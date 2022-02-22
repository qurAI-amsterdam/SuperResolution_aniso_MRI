import torch
from kwatsch.brain.trainer_ae import AETrainerExtension1Brain
from torch import distributions as distns
import torch.nn.functional as F


class VAETrainer(AETrainerExtension1Brain):

    def __init__(self, args, ae, max_grad_norm=0, model_file=None, eval_mode=False, **kwargs):
        super(VAETrainer, self).__init__(args, ae, max_grad_norm, model_file, eval_mode, **kwargs)
        self.recon_loss = lambda x_in, y_target: F.binary_cross_entropy(x_in, y_target, size_average=False). \
            div(x_in.size(0))
        self.train_combined = True if 'combined' in self.args['model'] else False
        self.dist_normal = None
        self.lamb = self.args['lamb']
        self.beta = self.args['vae_beta']
        if self.model.use_multiple_gpu and not eval_mode:
            print("INFO - VAETrainer - encoder_mu/log_var on second GPU!")
            self.model.encoder_mu = self.model.encoder_mu.to('cuda:1')
            # self.model.encoder_logvar = self.model.encoder_logvar.to('cuda:1')
        print("IMPORTANT --> {} training with combined losses: {}".format(self.__class__.__name__,
                                                                          self.train_combined))

    def train(self, batch_item, keep_predictions=True, eval_mode=False):
        x = batch_item['image'].to(self.args['device'])
        if not eval_mode:
            self.model.train()
        else:
            self.model.eval()

        self._iters += 1
        enc = self.model.encode(x)
        enc_logvar = self.model.encoder_logvar(enc)
        if self.model.use_multiple_gpu:
            enc = enc.to('cuda:1')
        enc_mu = self.model.encoder_mu(enc)

        if self.use_multiple_gpu:
            enc_mu = enc_mu.to('cuda:0')
            # enc_logvar = enc_logvar.to('cuda:0')
        z_sampled, dist_normal = self.sample_from_normal(enc_mu, enc_logvar)
        z_sampled = self.unflatten(z_sampled)
        out = self.model.decode(z_sampled)

        loss_dict = self.get_loss(x, out, is_test=False)
        kl_loss = self.get_kl_loss(dist_normal, enc_mu, enc_logvar)
        # print("Loss devices ", loss_dict['loss_ae'].device, kl_loss.device, x.device, out.device)
        loss_ae = self.lamb * loss_dict['loss_ae'] + self.beta * kl_loss

        slice_between = batch_item['slice_between'].to(z_sampled.device)
        # Important: is_eval must be FALSE in case we use combined loss!
        return_dict_sbi = self.synthesize_batch_images(batch_item=batch_item, z=z_sampled, compute_latent_loss=True,
                                                       slice_between=slice_between,
                                                       is_eval=False if self.train_combined else True)
        loss_ae_extra = self.get_extra_loss(slice_between, return_dict_sbi['s_between_mix'], return_dict_sbi['z_mix'],
                                            z=z_sampled, mask=None if not self.args['get_masks'] else batch_item['loss_mask'],
                                            is_test=False)
        if self.train_combined:
            if self.use_multiple_gpu:
                loss_ae_extra = loss_ae_extra.to('cuda:0')
            loss_ae = loss_ae + loss_ae_extra
        self.opt_ae.zero_grad()
        if not eval_mode:
            loss_ae.backward()
            self.opt_ae.step()

        # if lr_scheduler is not None take a scheduling step
        if self.opt_sched_ae is not None:
            self.opt_sched_ae.step()

        self.losses['loss_ae'].append(loss_ae.item())
        self.losses['loss_latent_1'].append(return_dict_sbi['loss_latent'].item())
        self.losses['kl_loss'].append(self.beta * kl_loss.item())
        slice_inbetween_mix = return_dict_sbi['s_between_mix'].detach().cpu()
        if keep_predictions:
            self.train_predictions = {'z_mix': return_dict_sbi['z_mix'].detach().cpu(),
                                      'pred_alphas': batch_item['alpha_from'].detach(),
                                      'slice_inbetween_mix': slice_inbetween_mix,
                                      'slice_inbetween_05': slice_inbetween_mix,
                                      "reconstruction": out.detach().cpu()}

    def get_loss(self, reference, recons, is_test=False, store_loss=True):
        loss_ae_dist = self.recon_loss(recons, reference)
        if self.use_multiple_gpu:
            loss_ae_dist = loss_ae_dist.to(device=reference.device)
        if store_loss:
            if is_test:
                self.losses_test['loss_ae_dist'].append(loss_ae_dist.item())
            else:
                self.losses['loss_ae_dist'].append(loss_ae_dist.item())

        return {"loss_ae": loss_ae_dist, 'loss_ae_dist': loss_ae_dist,
                "loss_laploss": 0}

    def get_kl_loss(self, dist_normal, enc_mu, enc_var):
        prior = distns.normal.Normal(torch.zeros_like(enc_mu), torch.ones_like(enc_var))
        kl_loss =  distns.kl.kl_divergence(dist_normal, prior).mean()
        return kl_loss

    def sample_from_normal(self, enc_mu, enc_logvar):
        if self.model.use_multiple_gpu:
            enc_mu = enc_mu.to('cuda:0')

        enc_sigma = enc_logvar.mul(0.5).exp_()
        distn = distns.normal.Normal(enc_mu, enc_sigma)
        enc_sampled = distn.rsample()
        return enc_sampled, distn

    def unflatten(self, t_tensor):
        return self.model.unflatten(t_tensor)

    def encode(self, x, eval=True, clear_cache=False, chunk_size=16, **kwargs):
        model = self._use_sr_model(kwargs.get("use_sr_model", False), origin="*encode*")
        if eval:
            model.eval()
            if clear_cache:
                torch.cuda.empty_cache()
        else:
            model.train()
        b, c, w, h = x.shape
        if w == 256 and h == 256 and b > chunk_size:
            # print("Warning - Trainer - Encoding - Chunking!!! {}".format(b))
            self.do_chunk = True
            if clear_cache:
                torch.cuda.empty_cache()
            x = x.detach().cpu()
            # we need to chunk
            if b % chunk_size == 0:
                chunks = b // chunk_size
            else:
                chunks = (b // chunk_size) + 1
            out = None
            for x_chunk in torch.chunk(x, chunks=chunks, dim=0):
                x_chunk = x_chunk.to('cuda')
                if eval:
                    with torch.no_grad():
                        out_x = model.encode(x_chunk)
                else:
                    out_x = model.encode(x_chunk)
                self.exec_vae_bottleneck(out_x, do_sample=False)
                out = torch.cat([out, out_x.detach().cpu()], dim=0) if out is not None else out_x.detach().cpu()
            return out
        if eval:
            with torch.no_grad():
                out = model.encode(x)
        else:
            out = model.encode(x)
        out = self.exec_vae_bottleneck(out, do_sample=False)
        return out

    def decode(self, z, eval=True, clear_cache=False, chunk_size=16, **kwargs):
        model = self._use_sr_model(kwargs.get("use_sr_model", False), origin="*decode*")
        if eval:
            model.eval()
            if clear_cache:
                torch.cuda.empty_cache()
        else:
            model.train()
        if z.dim() == 4:
            b, c, w, h = z.shape
        elif z.dim() == 2:
            b, _ = z.shape
        else:
            raise ValueError("Error - z has unsupported #dims {}".format(z.dim()))
        if self.do_chunk:
            # print("Warning - Trainer - Decoding - Chunking!!!")
            if clear_cache:
                torch.cuda.empty_cache()
            z = z.detach().cpu()
            # we need to chunk
            if b % chunk_size == 0:
                chunks = b // chunk_size
            else:
                chunks = (b // chunk_size) + 1
            out = None
            for z_chunk in torch.chunk(z, chunks=chunks, dim=0):
                z_chunk = z_chunk.to('cuda')
                if eval:
                    with torch.no_grad():
                        out_x = model.decode(z_chunk)
                else:
                    out_x = model.decode(z_chunk)
                out = torch.cat([out, out_x], dim=0) if out is not None else out_x
            self.do_chunk = False
            return out
        if not z.is_cuda:
            z = z.to('cuda')
        if eval:
            with torch.no_grad():
                out = model.decode(z)
        else:
            out = model.decode(z)
        return out

    def _validate_chunks(self, validation_batch, chunk_size=8):
        val_images = validation_batch['image']
        with torch.no_grad():
            z = self.encode(val_images, eval=True, clear_cache=False, chunk_size=16)
            out_x = self.decode(z, eval=True, clear_cache=False, chunk_size=16)

        return out_x, z

    def exec_vae_bottleneck(self, encodings, do_sample=False):
        if eval:
            with torch.no_grad():
                enc_logvar = self.model.encoder_logvar(encodings)
                if self.model.use_multiple_gpu:
                    # print("WARNING - exec_vae_bottleneck exec_vae_bottleneck MOVE ENCODINGS")
                    encodings = encodings.to('cuda:1')
                enc_mu = self.model.encoder_mu(encodings)

        else:
            enc_logvar = self.model.encoder_logvar(encodings)
            if self.model.use_multiple_gpu:
                # print("WARNING - exec_vae_bottleneck exec_vae_bottleneck MOVE ENCODINGS")
                encodings = encodings.to('cuda:1')
            enc_mu = self.model.encoder_mu(encodings)

        if do_sample:
            z, dist_normal = self.sample_from_normal(enc_mu, enc_logvar)
        else:
            z = enc_mu
        if self.use_multiple_gpu:
            z = z.to('cuda:0')
        return self.unflatten(z)


class VAECombinedTrainer(AETrainerExtension1Brain):

    def get_extra_loss(self, slice_between, s_between_mix, z_mix, z=None, mask=None, is_test=False):
        if self.args['use_loss_annealing']:
            # print("Loss weight {:.6f}".format(self.loss_weights[self.epoch]))
            loss_extra_image = self.loss_weights[self.epoch] * self.get_extra_image_loss(slice_between, s_between_mix,
                                                                                         mask=mask)
        else:
            loss_extra_image = self.args['ex_loss_weight1'] * self.get_extra_image_loss(slice_between, s_between_mix,
                                                                                        mask=mask)
        if self.args['use_extra_latent_loss']:
            loss_extra_latent = 0.5 * self.get_extra_latent_loss(s_between_mix, slice_between, z_mix, is_test=is_test,
                                                                 z=z)
            loss_ae_extra = loss_extra_latent + loss_extra_image
        else:
            loss_ae_extra = loss_extra_image

        if is_test:
            # self.losses_test['loss_ae_extra'].append(loss_ae_extra.item())
            self.losses_test['loss_ae_dist_extra'].append(loss_extra_image.item())
        else:
            # self.losses['loss_ae_extra'].append(loss_ae_extra.item())
            self.losses['loss_ae_dist_extra'].append(loss_extra_image.item())
        return loss_ae_extra