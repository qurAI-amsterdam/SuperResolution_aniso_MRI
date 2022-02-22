import os
import numpy as np
from kwatsch.trainer_ae import AEBaseTrainer
import torch
import torch.nn.functional as F


class AEBaseTrainerBrain(AEBaseTrainer):

    def _get_mixup_image(self, **kwargs):
        z, is_test = kwargs.get('z'), kwargs.get('is_test', False)
        z_mix = self._get_mixup_latent(**kwargs)
        if is_test:
            with torch.no_grad():
                s_slice_mix = self.model.decode(z_mix)
        else:
            s_slice_mix = self.model.decode(z_mix)
        return {'z_mix': z_mix, 'slice_inbetween_mix': s_slice_mix}

    def _get_mixup_latent(self, **kwargs):
        z, alpha_from, alpha_to = kwargs.get('z'), kwargs.get('alpha_from'), kwargs.get('alpha_to')
        # print("get mixup latent ", z.shape, alpha_from.shape, alpha_to.shape, z.dtype, alpha_to.dtype)
        if z.dim() == 4:
            z_mix = alpha_from[:, :, None, None].to(z.device) * z[:z.size(0) // 2] + \
                    alpha_to[:, :, None, None].to(z.device) * z[z.size(0) // 2:]
        elif z.dim() == 2:
            z_mix = alpha_from.to(z.device) * z[:z.size(0) // 2] + \
                    alpha_to.to(z.device) * z[z.size(0) // 2:]
        else:
            raise ValueError("Insupported shape of encodings z in _get_mixup_latent!")

        return z_mix

    def get_latent_loss(self, **kwargs):
        reference, z, no_grad = kwargs.get('reference'), kwargs.get('z'), kwargs.get('no_grad', False)
        z_mix = self._get_mixup_latent(**kwargs)
        if no_grad:
            with torch.no_grad():
                z_ref = self.encode(reference, eval=True)
        else:
            z_ref = self.encode(reference, eval=False)
        latent_loss = F.mse_loss(z_mix if z_ref.is_cuda else z_mix.detach().cpu(), z_ref)
        # latent_loss = (1 - F.cosine_similarity(z_mix if z_ref.is_cuda else z_mix.detach().cpu(), z_ref)).mean()
        return {'loss_latent': latent_loss, 'z_mix': z_mix}


class AETrainerBrain(AEBaseTrainerBrain):

    def train(self, batch_item, keep_predictions=True, eval_mode=False):
        x = batch_item['image'].to(self.args['device'])
        if not eval_mode:
            self.model.train()
        else:
            self.model.eval()

        self._iters += 1
        z = self.model.encode(x)
        out = self.model.decode(z)
        if self.use_multiple_gpu:
            out = out.to('cuda:1')
            x = x.to('cuda:1')

        loss_ae = self.get_loss(x, out, is_test=False)['loss_ae']
        return_dict = self.get_latent_loss(reference=batch_item['slice_between'].to(z.device),
                                           alpha_from=batch_item['alpha_from'],
                                           alpha_to=batch_item['alpha_to'],
                                           z=z, no_grad=True)
        self.opt_ae.zero_grad()
        if not eval_mode:
            loss_ae.backward()
            self.opt_ae.step()

        # if lr_scheduler is not None take a scheduling step
        if self.opt_sched_ae is not None:
            self.opt_sched_ae.step()

        self.losses['loss_ae'].append(loss_ae.item())
        self.losses['loss_latent_1'].append(return_dict['loss_latent'].item())
        mix_result_dict = self._get_mixup_image(z=z, alpha_from=batch_item['alpha_from'],
                                                alpha_to=batch_item['alpha_to'], is_test=True)
        slice_inbetween_mix = mix_result_dict['slice_inbetween_mix'].detach().cpu()
        if keep_predictions:
            self.train_predictions = {'z_mix': return_dict['z_mix'].detach().cpu(),
                                      'pred_alphas': batch_item['alpha_from'].detach(),
                                      'slice_inbetween_mix': slice_inbetween_mix,
                                      'slice_inbetween_05': slice_inbetween_mix,
                                      "reconstruction": out.detach().cpu()}


class AETrainerExtension1Brain(AEBaseTrainerBrain):

    def train(self, batch_item, keep_predictions=True, eval_mode=False):
        x = batch_item['image'].to(self.args['device'])
        if not eval_mode:
            self.model.train()
        else:
            self.model.eval()

        self._iters += 1
        z = self.model.encode(x)
        out = self.model.decode(z)
        if self.use_multiple_gpu:
            out = out.to('cuda:1')
            x = x.to('cuda:1')

        loss_ae = self.get_loss(x, out, is_test=False)['loss_ae']
        slice_between = batch_item['slice_between'].to(z.device)
        return_dict_sbi = self.synthesize_batch_images(batch_item=batch_item, z=z, compute_latent_loss=True,
                                                       slice_between=slice_between, is_eval=False)
        loss_ae_extra = self.get_extra_loss(slice_between, return_dict_sbi['s_between_mix'], return_dict_sbi['z_mix'],
                                            z=z, mask=None if not self.args['get_masks'] else batch_item['loss_mask'],
                                            is_test=False)
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
        if keep_predictions:
            s_between_mix = return_dict_sbi['s_between_mix'].detach().cpu()
            self.train_predictions = {'z_mix': return_dict_sbi['z_mix'].detach().cpu(),
                                      'pred_alphas': torch.FloatTensor([0.5]),
                                      'slice_inbetween_mix': s_between_mix,
                                      'slice_inbetween_05': s_between_mix,
                                      "reconstruction": out.detach().cpu()}

    def validate(self, validation_batch, image_dict=None, frame_id=8, generate_images=True):
        val_result_dict = super().validate(validation_batch, image_dict=image_dict, frame_id=frame_id,
                                           generate_images=generate_images)

        z = self.test_predictions['z'].to(self.test_predictions['z_device'])
        slice_between = validation_batch['slice_between'].to(z.device)
        return_dict_sbi = self.synthesize_batch_images(batch_item=validation_batch, z=z, compute_latent_loss=True,
                                                       slice_between=slice_between, is_eval=True)
        self.losses_test['loss_latent_1'].append(return_dict_sbi['loss_latent'].item())
        loss_ae_extra = self.get_extra_loss(slice_between, return_dict_sbi['s_between_mix'], return_dict_sbi['z_mix'],
                                               mask=None if not self.args['get_masks'] else validation_batch['loss_mask'],
                                               is_test=True)
        self.losses_test['loss_ae'][-1] = self.losses_test['loss_ae'][-1] + loss_ae_extra.item()
        if self.epoch > self.args['epoch_threshold']:
            # print("INFO - BaseTrainer - saved models e{} > {}".format(self.epoch, self.args['epoch_threshold']))
            self.save_best_val_model()

        return val_result_dict

    def save_best_val_model(self, **kwargs):
        super().save_best_val_model()
        if len(self.mean_losses_test['loss_ae_dist_extra']) > 1:
            if np.argmin(self.mean_losses_test['loss_ae_dist_extra']) + 1 == len(self.mean_losses_test['loss_ae_dist_extra']):
                # the last loss is the lowest, save model
                # although slightly awkward (otherwise have to adjust too much other stuff,
                # we pass self.epoch which is initialized by 0, therefore plus 1
                fname = os.path.join(self.args['dir_models'], 'caisr.models')
                self.save_models(fname, self.epoch + 1)

    def get_extra_loss(self, slice_between, s_between_mix, z_mix, z=None, mask=None, is_test=False):
        loss_extra_image = self.args['ex_loss_weight1'] * self.get_extra_image_loss(slice_between, s_between_mix,
                                                                                    mask=mask, is_test=is_test)
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

    def get_extra_image_loss(self, reference, synthesized, mask=None, is_test=False):

        if self.percept_criterion is not None and self.image_mix_loss_func == 'perceptual':

            if self.use_multiple_gpu:
                reference, synthesized = reference.to('cuda:1'), synthesized.to('cuda:1')
            if self.args['get_masks'] and mask is not None:
                if is_test:
                    with torch.no_grad():
                        loss_image = self.percept_criterion(reference, synthesized, normalize=True)
                        loss_image = torch.mean(
                            loss_image * mask[:loss_image.size(0)].to(loss_image.device))
                else:
                    loss_image = self.percept_criterion(reference, synthesized, normalize=True)
                    loss_image = torch.mean(
                        loss_image * mask[:loss_image.size(0)].to(loss_image.device))
            else:
                if is_test:
                    if reference.shape[0] > 16:
                        loss_image = self._chunk_perceptual_loss(reference, synthesized)
                    else:
                        with torch.no_grad():
                            loss_image = self.percept_criterion(reference, synthesized, normalize=True).mean()
                else:
                    loss_image = self.percept_criterion(reference, synthesized, normalize=True).mean()
        else:
            if self.use_multiple_gpu:
                reference, synthesized = reference.to('cuda:1'), synthesized.to('cuda:1')
            if self.args['get_masks'] and mask is not None:
                loss_image = F.mse_loss(reference, synthesized, reduction='none')
                loss_image = torch.mean(
                    loss_image * mask[:loss_image.size(0)].to(loss_image.device))
            else:
                # loss_image = F.smooth_l1_loss(synthesized, reference, reduction='mean')
                loss_image = F.mse_loss(reference, synthesized)
                # loss_image = 1 - self.ssim_criterion(synthesized, reference)
            if self.laploss is not None:
                if is_test:
                    with torch.no_grad():
                        loss_image = self.laploss(synthesized, reference) + loss_image
                else:
                    loss_image = self.laploss(synthesized, reference) + loss_image

            if self.use_multiple_gpu:
                loss_image = loss_image.to('cuda:1')

        return loss_image

    def get_extra_latent_loss(self, img_synthesized, img_reference, z_synthesized, device=None, is_test=False, z=None):
        # img_synthesized: decoding of the predicted latent mix for the slice-in-between
        # img_reference: the corresponding reference (original) for the img_synthesized (in-between slice)
        # we detach z_reference (slice in between encoding), because we don't want the loss caused by
        # alpha prediction to be accounted for by the encoder
        if is_test:
            self.model.eval()
            with torch.no_grad():
                z_reference = self.model.encode(img_reference)
                z_mix_pred = self.model.encode(img_synthesized)
        else:
            z_reference = self.model.encode(img_reference)
            z_mix_pred = self.model.encode(img_synthesized)
        z_loss = F.mse_loss(z_reference, z_synthesized)
        z_mix_loss = F.mse_loss(z_reference, z_mix_pred)
        if self.use_multiple_gpu and device is not None:
            z_loss, z_mix_loss = z_loss.to(device), z_mix_loss.to(device)
        if is_test:
            self.losses_test["loss_latent_1"].append(z_loss.item())
            self.losses_test["loss_latent_2"].append(z_mix_loss.item())
            self.model.train()
        else:
            self.losses["loss_latent_1"].append(z_loss.item())
            self.losses["loss_latent_2"].append(z_mix_loss.item())
        return z_loss + z_mix_loss

    def synthesize_batch_images(self, **kwargs):
        batch_item, z, slice_between = kwargs.get('batch_item'), kwargs.get('z'), kwargs.get('slice_between', None)
        is_eval, compute_latent_loss = kwargs.get('is_eval', False), kwargs.get('compute_latent_loss', False)
        latent_loss = 0
        if is_eval:
            self.model.eval()
            if not z.is_cuda:
                z = z.to('cuda:0')

        if z.dim() == 4:
            z_mix = batch_item['alpha_from'][:, :, None, None].to(z.device) * z[:z.size(0) // 2] + \
                    batch_item['alpha_to'][:, :, None, None].to(z.device) * z[z.size(0) // 2:]
        elif z.dim() == 2:
            z_mix = batch_item['alpha_from'].to(z.device) * z[:z.size(0) // 2] + \
                    batch_item['alpha_to'].to(z.device) * z[z.size(0) // 2:]
        else:
            raise ValueError("Insupported shape of encodings z in sysnthesize_batch_images!")

        if not is_eval:
            s_between_mix = self.model.decode(z_mix)
        else:
            s_between_mix = self.decode(z_mix, eval=is_eval, clear_cache=False)
            self.model.train()
        if compute_latent_loss:
            z_ref = self.encode(slice_between, eval=is_eval)
            latent_loss = F.mse_loss(z_mix if z_ref.is_cuda else z_mix.detach().cpu(), z_ref)
        return {'s_between_mix': s_between_mix, 'z_mix': z_mix, 'loss_latent': latent_loss}

