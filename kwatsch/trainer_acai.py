import numpy as np
from torch import optim
from networks.acai_vanilla import Discriminator
from kwatsch.brain.trainer_ae import AETrainerExtension1Brain
import torch
import torch.nn.functional as F
from networks.acai_vanilla import swap_halves


def swap_halves(x):
    a, b = x.split(x.shape[0]//2)
    return torch.cat([b, a])


# torch.lerp only support scalar weight
def lerp(start, end, weights):
    return start + weights * (end - start)


def interp_image(image1, image2, alpha):
    """
    assuming both images are torch tensors

    :param image1:
    :param image2:
    :param alpha: scalar
    :return:
    """
    return alpha * image1 + ((1 - alpha) * image2)


class ACAITrainer(AETrainerExtension1Brain):

    def __init__(self, args, ae, max_grad_norm=0, model_file=None, eval_mode=False, **kwargs):
        super(ACAITrainer, self).__init__(args, ae, max_grad_norm, model_file, eval_mode, **kwargs)
        self.train_combined = True if 'combined' in self.args['model'] else False
        self.dist_normal = None
        self.disc_model = None
        self._get_discriminator()
        self.opt_disc = optim.Adam(self.disc_model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'],
                                   betas=(0.9, 0.999))
        self.gamma_reg_acai = 0.2  # hyperparameters as used in Berthelot et al. paper
        self.z = None
        print("IMPORTANT --> {} training with combined losses: {}".format(self.__class__.__name__,
                                                                          self.train_combined))

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

        loss_ae_dist = self.get_loss(x, out, is_test=False)['loss_ae_dist']
        # Important, reconstruction, reference sequence
        loss_dict = self.get_loss_disc(out, x, z, is_test=False)
        loss_ae = loss_ae_dist + self.args['lamb_reg_acai'] * loss_dict['loss_ae_l2']
        loss_disc = loss_dict['loss_disc_dist'] + loss_dict['loss_disc_l2']
        slice_between = batch_item['slice_between'].to(z.device)
        return_dict_sbi = self.synthesize_batch_images(batch_item=batch_item, z=z, compute_latent_loss=True,
                                                       slice_between=slice_between,
                                                       is_eval=False if self.train_combined else True)
        loss_ae_extra = self.get_extra_loss(slice_between, return_dict_sbi['s_between_mix'], return_dict_sbi['z_mix'],
                                            z=z, mask=None if not self.args['get_masks'] else batch_item['loss_mask'],
                                            is_test=False if self.train_combined else True)
        if self.train_combined:
            loss_ae = loss_ae + loss_ae_extra

        if not eval_mode:
            self.opt_ae.zero_grad()
            self.opt_disc.zero_grad()
            loss_ae.backward(retain_graph=True)
            loss_disc.backward()
            self.opt_ae.step()
            self.opt_disc.step()

        # if lr_scheduler is not None take a scheduling step
        if self.opt_sched_ae is not None:
            self.opt_sched_ae.step()

        self.losses['loss_ae'].append(loss_ae.item())
        self.losses['loss_latent_1'].append(return_dict_sbi['loss_latent'].item())
        self.losses['loss_disc'].append(loss_disc.item())
        # print("Loss ", self.losses['loss_ae'][-1], self.losses['loss_disc'][-1])
        slice_inbetween_mix = return_dict_sbi['s_between_mix'].detach().cpu()
        if keep_predictions:
            self.train_predictions = {'z_mix': return_dict_sbi['z_mix'].detach().cpu(),
                                      'pred_alphas': batch_item['alpha_from'].detach(),
                                      'slice_inbetween_mix': slice_inbetween_mix,
                                      'slice_inbetween_05': slice_inbetween_mix,
                                      "reconstruction": out.detach().cpu()}

    def get_loss_disc(self, reconstruction, reference, z, is_test=True):
        # disc_mix_reg = torch.lerp(reconstruction, reference, self.gamma_reg_acai)
        disc_mix_reg = reconstruction + self.gamma_reg_acai * (reference - reconstruction)
        if is_test:
            with torch.no_grad():
                loss_disc_l2 = torch.mean(self.disc_model(disc_mix_reg)**2)
        else:
            loss_disc_l2 = torch.mean(self.disc_model(disc_mix_reg)**2)
        alpha = torch.rand(z.size(0) // 2, 1, 1, 1).to(self.args['device']) / 2
        z_mix = alpha * z[:z.size(0) // 2] + (1 - alpha) * z[z.size(0) // 2:]
        # Note: if batch-size is 32, in prepare_batch_pairs_brain (main) the batch is doubled with adjacent slices
        # if we then apply mixing alpha * z[:z.size(0) // 2] + (1 - alpha) * z[z.size(0) // 2:] we should get slices
        # close to the real once in-between. But to train ACAI we make the mixes more rough in the hope the model
        # that encourages the model to do better interpolation...
        # alpha = torch.rand(z.size(0), 1, 1, 1).to(self.args['device']) / 2
        # z_mix = lerp(z, swap_halves(z), alpha)
        out_mix = self.model.decode(z_mix)
        if self.use_multiple_gpu:
            out_mix = out_mix.to('cuda:1')
        disc_mix = self.disc_model(out_mix)
        loss_ae_l2 = torch.mean(disc_mix**2)
        loss_disc_dist = F.mse_loss(disc_mix, alpha.reshape(-1), reduction='mean')
        # loss_ae = loss_ae_dist + loss_ae_l2
        # loss_disc = loss_disc_dist + loss_disc_l2
        return {'loss_disc_l2': loss_disc_l2, 'loss_ae_l2': loss_ae_l2,
                'loss_disc_dist': loss_disc_dist}

    # def synthesize_batch_images(self, **kwargs):
    #     batch_item, z, slice_between = kwargs.get('batch_item'), kwargs.get('z'), kwargs.get('slice_between', None)
    #     is_eval, compute_latent_loss = kwargs.get('is_eval', False), kwargs.get('compute_latent_loss', False)
    #     latent_loss = 0
    #     if is_eval:
    #         self.model.eval()
    #         if not z.is_cuda:
    #             z = z.to('cuda:0')
    #     z_mix = self.alpha05 * z[:z.size(0) // 2] + (1 - self.alpha05) * z[z.size(0) // 2:]
    #     if not is_eval:
    #         s_between_mix = self.model.decode(z_mix)
    #     else:
    #         s_between_mix = self.decode(z_mix, eval=is_eval, clear_cache=False)
    #         self.model.train()
    #     if compute_latent_loss:
    #         z_ref = self.encode(slice_between, eval=is_eval)
    #         latent_loss = F.mse_loss(z_mix if z_ref.is_cuda else z_mix.detach().cpu(), z_ref)
    #     return {'s_between_mix': s_between_mix, 'z_mix': z_mix, 'loss_latent': latent_loss}

    def _get_discriminator(self):
        self.disc_model = Discriminator(self.args)

        if self.use_multiple_gpu:
            self.disc_model = self.disc_model.to('cuda:1')
        else:
            self.disc_model = self.disc_model.to(self.args['device'])
        print("INFO - Trainer ACAI - Initiated discriminator")

    def save_models(self, fname, epoch):
        torch.save({'model_dict_ae': self.model.state_dict(),
                    'optimizer_dict_ae': self.opt_ae.state_dict(),
                    'model_disc': self.disc_model.state_dict(),
                    'optimizer_disc': self.opt_disc.state_dict(),
                    'epoch': epoch}, fname)
