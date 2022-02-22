import os
import torch
from collections import defaultdict
import torch.nn.functional as F
from kwatsch.base_trainer import BaseTrainer
import numpy as np
from kwatsch.training_utils import save_image_grid
from kwatsch.acai_utils import generate_recon_grid
from torchvision.utils import make_grid


class AlphaBaseTrainer(BaseTrainer):

    def get_extra_loss(self, slice_between, s_between_mix, z_mix, z=None, mask=None, is_test=False):
        loss_extra_image = self.get_extra_image_loss(slice_between, s_between_mix,
                                                     mask=mask)
        loss_extra_latent = self.get_latent_loss(s_between_mix, slice_between, z_mix, device=loss_extra_image.device,
                                                 is_test=is_test, z=z)
        loss_ae_extra = 0.5 * loss_extra_image + 0.5 * loss_extra_latent
        # loss_ae_extra = loss_extra_latent
        # loss_ae_extra = loss_extra_image
        if is_test:
            self.losses_test['loss_ae_extra'].append(loss_ae_extra.item())
            self.losses_test['loss_ae_dist_extra'].append(loss_extra_image.item())
        else:
            self.losses['loss_ae_extra'].append(loss_ae_extra.item())
            self.losses['loss_ae_dist_extra'].append(loss_extra_image.item())
        return loss_ae_extra

    def get_latent_loss(self, img_synthesized, img_reference, z_synthesized, device=None, is_test=False, z=None):
        # img_synthesized: decoding of the predicted latent mix for the slice-in-between
        # img_reference: the corresponding reference (original) for the img_synthesized (in-between slice)
        # we detach z_reference (slice in between encoding), because we don't want the loss caused by
        # alpha prediction to be accounted for by the encoder
        if is_test:
            self.model.eval()
            self.alpha_probe.eval()
        with torch.no_grad():
            z_reference = self.model.encode(img_reference)
            z_mix_pred = self.model.encode(img_synthesized)
        z_loss = F.mse_loss(z_reference, z_synthesized)
        if z is not None:
            z_05 = self._get_mixup_latent(z)
            z_loss_05 = F.mse_loss(z_reference, z_05)
        z_mix_loss = F.mse_loss(z_reference, z_mix_pred)  # added detach
        if self.use_multiple_gpu and device is not None:
            z_loss, z_mix_loss = z_loss.to(device), z_mix_loss.to(device)
        if is_test:
            self.losses_test["loss_latent_1"].append(z_loss.item())
            self.losses_test["loss_latent_2"].append(z_mix_loss.item())
            if z is not None:
                self.losses_test["loss_latent_05"].append(z_loss_05.item())
            self.model.train()
            self.alpha_probe.train()
        else:
            self.losses["loss_latent_1"].append(z_loss.item())
            self.losses["loss_latent_2"].append(z_mix_loss.item())
            if z is not None:
                self.losses["loss_latent_05"].append(z_loss_05.item())
        return z_loss + z_mix_loss

    def get_extra_image_loss(self, reference, synthesized, mask=None):

        if self.percept_criterion is not None and self.alpha_loss_func == 'perceptual':
            if self.use_multiple_gpu:
                reference, synthesized = reference.to('cuda:1'), synthesized.to('cuda:1')
                # s_between_recon = s_between_recon.to('cuda:1')
            if self.args['get_masks'] and mask is not None:
                loss_image = self.percept_criterion(reference, synthesized, normalize=True)
                loss_image = torch.mean(
                    loss_image * mask[:loss_image.size(0)].to(loss_image.device))
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
                loss_image = F.mse_loss(reference, synthesized)
            if self.laploss is not None:
                loss_image = self.laploss(synthesized, reference) + loss_image

            if self.use_multiple_gpu:
                loss_image = loss_image.to('cuda:1')

        return loss_image

    def predict_alpha(self, combined_z, feature_map, eval=True):
        if eval:
            self.alpha_probe.eval()
        else:
            self.alpha_probe.train()
        return self.alpha_probe(combined_z, feature_map)

    def synthesize_batch_images(self, batch_item, z, is_eval=False):

        feature_map = self.create_add_features(batch_item, z.size(0) // 2, z.device)
        z1, z3 = z.split(z.size(0) // 2, dim=0)
        z_reshape = torch.cat([z1, z3], dim=1)
        if is_eval:
            self.alpha_probe.eval()
            self.model.eval()
        # we detach z_reshape: means does not contribute to loss caused by alpha prediction
        alpha = self.alpha_probe(z_reshape.detach(), feature_map)
        if self.num_alphas == 16:
            z_mix = alpha[:, :, None, None] * z1 + (1 - alpha[:, :, None, None]) * z3
        elif self.num_alphas == 32:
            latent_dim = int(self.args['latent'])
            z_mix = alpha[:, :latent_dim, None, None] * z1 + alpha[:, latent_dim:, None, None] * z3
        elif self.num_alphas == 256:
            alpha = alpha.view(z1.size(0), 1, z1.size(2), z1.size(3))
            z_mix = alpha * z1 + (1 - alpha) * z3
        else:
            raise ValueError("Error - synthesize_batch_images - unknown parameter num_alphas {}".format(self.num_alphas))
            # z_mix = alpha[:, 0, None, None, None] * z1 + alpha[:, 1, None, None, None] * z3
        # we decode the synthesized latent code WITHOUT computing gradients. We don't want decoder weights
        # to contribute to the loss of the alpha prediction
        with torch.no_grad():
            s_between_mix = self.model.decode(z_mix)
        if is_eval:
            self.alpha_probe.train()
            self.model.train()
        return s_between_mix, alpha, z_mix

    def validate(self, validation_batch, image_dict=None, frame_id=8):
        self.alpha_probe.eval()
        val_result_dict = super().validate(validation_batch, image_dict=image_dict,
                                                               frame_id=frame_id)
        z = self.test_predictions['z'].to(self.test_predictions['z_device'])
        slice_between = validation_batch['slice_between'].to(z.device)
        s_between_mix, pred_alphas, z_mix = self.synthesize_batch_images(validation_batch, z, is_eval=True)
        _ = self.get_extra_loss(slice_between, s_between_mix, z_mix,
                                               mask=None if not self.args['get_masks'] else validation_batch['loss_mask'],
                                               is_test=True)

        return val_result_dict

    def _determine_num_alphas(self):
        # print("INFO - AlphaTrainer - loading class {}".format(self.alpha_probe.__class__.__name__))
        if self.alpha_probe.__class__.__name__ == "AlphaProbe16Convex":
            self.num_alphas = 16
        elif self.alpha_probe.__class__.__name__ in ["AlphaProbe16v1", "AlphaProbe16v2"]:
            self.num_alphas = 32

        elif self.alpha_probe.__class__.__name__[:13] == "AlphaProbe256":
            self.num_alphas = 256
        else:
            self.num_alphas = 2
        print("WARNING - {} - AlphaProbe is predicting {} coefficients!".format(self.alpha_probe.__class__.__name__,
                                                                                self.num_alphas))

    def determine_alpha_loss_func(self):
        if "alpha_loss_func" in self.args.keys():
            self.alpha_loss_func = self.args['alpha_loss_func']
        else:
            if self.args['use_percept_loss']:
                self.alpha_loss_func = "perceptual"
            else:
                self.alpha_loss_func = "mse"

    def end_epoch_processing(self, **kwargs):
        epoch = kwargs.get('epoch')
        val_result_dict = kwargs.get('val_result_dict')
        super().end_epoch_processing(**kwargs)
        # Save example validation volumes as .png to disc
        s1_alphas, s2_alphas = [], []
        fname = os.path.join(self.args['dir_images'], 'val_image_e{:03d}_xxx.png'.format(epoch))
        for p_id, s_img in val_result_dict['synthesized_vols'].items():
            save_image_grid(s_img * 255, filename=fname.replace('xxx', 'p{:03d}'.format(p_id)))
            s1_alphas.extend(val_result_dict['alphas'][p_id][:, 0].ravel())
            s2_alphas.extend(val_result_dict['alphas'][p_id][:, 1].ravel())
        self.add_hist_tensorboard(s1_alphas, "alphas/test-dim0")
        self.add_hist_tensorboard(s2_alphas, "alphas/test-dim1")

    @staticmethod
    def create_add_features(batch_dict, batch_size, device='cuda'):
        # z should have shape [b, latent, latent_w, latent_h]
        # we add info about slice1 and slice2 as well as frame_id: 3 features maps
        s_id1, s_id3, n_slices = batch_dict['slice_id_from'], batch_dict['slice_id_to'], batch_dict['num_slices_vol']
        s_spacing = batch_dict['spacing'][:batch_size, 0]
        ns_id1, ns_id3 = (s_id1[:batch_size] + 1) * 1/n_slices[:batch_size], (s_id3[:batch_size] + 1) * 1/n_slices[:batch_size]
        # ns_id1, ns_id3 = ns_id1.view(-1, 1, 1, 1).expand((b, 1, y, x)), ns_id3.view(-1, 1, 1, 1).expand((b, 1, y, x))
        f_id = (batch_dict['frame_id_from'][:batch_size] + 1) * 1/batch_dict['num_frames_vol'][:batch_size]

        return torch.cat([ns_id1.to(device), ns_id3.to(device), f_id.to(device),
                          s_spacing[:, None].to(device), n_slices[:batch_size].to(device)], dim=1)

    def load(self, fname):
        state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict['model_dict_ae'])
        self.opt_ae.load_state_dict(state_dict['optimizer_dict_ae'])
        if 'alpha_probe' not in state_dict.keys():
            print("Warning - {} - cannot load alpha probe network. Not in state dict!".format(self.__class__.__name__))
        else:
            self.alpha_probe.load_state_dict(state_dict['alpha_probe'])
        print("INFO - {} Loaded ae & alpha-probe model parameters from {}".format(self.__class__.__name__, fname))

    def save_models(self, fname, epoch):
        torch.save({'model_dict_ae': self.model.state_dict(),
                    'optimizer_dict_ae': self.opt_ae.state_dict(),
                    'alpha_probe': self.alpha_probe.state_dict(),
                    'epoch': epoch}, fname)

