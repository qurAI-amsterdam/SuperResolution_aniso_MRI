import os
import torch
from collections import defaultdict
import torch.nn.functional as F
from torch import optim
from kwatsch.common import load_settings
from kwatsch.base_trainer import BaseTrainer
from kwatsch.alpha.base_alpha_trainer import AlphaBaseTrainer
import numpy as np
from kwatsch.training_utils import save_image_grid
from kwatsch.acai_utils import generate_recon_grid
from torchvision.utils import make_grid


class AlphaOnlyTrainer(AlphaBaseTrainer):

    def __init__(self, args, ae, max_grad_norm=0, alpha_probe=None, model_file=None, eval_mode=False):
        super(AlphaOnlyTrainer, self).__init__()
        self.args = args
        self.model = ae
        self.load_ae_params(os.path.expanduser(args['model_file_ae']))
        self.model.eval()
        self.alpha_probe = alpha_probe
        self.eval_model = eval_mode
        momentum = 0.9 if 'momentum' not in args.keys() else args['momentum']
        self._determine_num_alphas()
        self.opt_ae = optim.Adam(self.alpha_probe.parameters(), lr=args['lr'], weight_decay=0, betas=(momentum, 0.999))
        self.opt_sched_ae = None
        self._init_scheduler()
        self.losses = defaultdict(list)
        self.losses_test = defaultdict(list)
        self.train_predictions, self.test_predictions = None, None
        self._iters = 1
        self.max_grad_norm = max_grad_norm
        if self.eval_model:
            print("AlphaOnlyTrainer - model is in eval mode ", self.eval_model)
        self.use_multiple_gpu = True if not self.eval_model and ('gpu_ids' in args.keys() and len(args['gpu_ids']) > 1) else False

        self._init_laploss()
        self._init_percept_loss()
        if model_file is not None:
            self.load(model_file)

    def train(self, batch_item, keep_predictions=True):
        x = batch_item['image'].to(self.args['device'])
        self._iters += 1
        self.alpha_probe.train()
        with torch.no_grad():
            z = self.model.encode(x)
            out = self.model.decode(z)
        if self.use_multiple_gpu:
            out = out.to('cuda:1')
            x = x.to('cuda:1')

        slice_between = batch_item['slice_between'].to(z.device)
        s_between_mix, pred_alphas, z_mix = self.synthesize_batch_images(batch_item, z)
        loss_ae_extra = self.get_extra_loss(slice_between, s_between_mix, z_mix, z=z,
                                               mask=None if not self.args['get_masks'] else batch_item['loss_mask'],
                                               is_test=False)
        loss_ae = loss_ae_extra

        self.opt_ae.zero_grad()
        loss_ae.backward()
        self.opt_ae.step()

        # if lr_scheduler is not None take a scheduling step
        if self.opt_sched_ae is not None:
            self.opt_sched_ae.step()

        self.losses['loss_ae'].append(loss_ae.item())

        if keep_predictions:
            self.train_predictions = {'z_mix':z_mix.detach().cpu(),
                                      'pred_alphas': pred_alphas.detach().cpu(),
                                      'slice_inbetween_mix': s_between_mix.detach().cpu(),
                                      'slice_inbetween_05': self._get_mixup_image(z=z).detach().cpu(),
                                      "reconstruction": out.detach().cpu()}

    def get_extra_loss(self, slice_between, s_between_mix, z_mix, z=None, mask=None, is_test=False):
        loss_extra_image = self.get_extra_image_loss(slice_between, s_between_mix,
                                                     mask=mask)
        loss_extra_latent = self.get_latent_loss(s_between_mix, slice_between, z_mix, device=loss_extra_image.device,
                                                 is_test=is_test, z=z)
        loss_ae_extra = 0.005 * loss_extra_image + 0.5 * loss_extra_latent
        # loss_ae_extra = loss_extra_latent
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

        if self.percept_criterion is not None:
            if self.use_multiple_gpu:
                slice_between, s_between_mix = reference.to('cuda:1'), synthesized.to('cuda:1')
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

    def _determine_num_alphas(self):
        print("INFO - AlphaOnlyTrainer - loading class {}".format(self.alpha_probe.__class__.__name__))
        if self.alpha_probe.__class__.__name__[:12] == "AlphaProbe16":

            self.num_alphas = 16
        elif self.alpha_probe.__class__.__name__[:13] == "AlphaProbe256":
            self.num_alphas = 256
        else:
            self.num_alphas = 2

    def load_ae_params(self, fname):
        state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict['model_dict_ae'])