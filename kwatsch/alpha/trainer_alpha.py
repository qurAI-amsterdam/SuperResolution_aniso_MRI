
import torch
from kwatsch.alpha.base_alpha_trainer import AlphaBaseTrainer
from collections import defaultdict
import torch.nn.functional as F
from torch import optim
from kwatsch.base_trainer import BaseTrainer
import numpy as np
from kwatsch.training_utils import save_image_grid
from kwatsch.acai_utils import generate_recon_grid
from torchvision.utils import make_grid


class AlphaTrainer(AlphaBaseTrainer):

    def __init__(self, args, ae, max_grad_norm=0, alpha_probe=None, model_file=None, eval_mode=False):
        super(AlphaTrainer, self).__init__()
        self.args = args
        self.model = ae
        self.alpha_probe = alpha_probe
        self.eval_model = eval_mode
        # This is only important for models predicting alpha coefficients. Used in create_super_volume function
        # When set to True, overrules alpha predictions and uses fixed coefficients e.g. [0.5]
        self.eval_fixed_coeff = False
        momentum = 0.9 if 'momentum' not in args.keys() else args['momentum']
        params = list(self.model.parameters()) + list(self.alpha_probe.parameters())
        # print("WARNING - AlphaTrainer - Batchnorm: {}".format(args['use_batchnorm_probe']))
        self._determine_num_alphas()
        self.opt_ae = optim.Adam(params, lr=args['lr'], weight_decay=0, betas=(momentum, 0.999))
        self.opt_sched_ae = None
        self._init_scheduler()
        self.losses = defaultdict(list)
        self.losses_test = defaultdict(list)
        self.train_predictions, self.test_predictions = None, None
        self._iters = 1
        self.max_grad_norm = max_grad_norm
        if self.eval_model:
            print("AlphaTrainer - model is in eval mode ", self.eval_model)
        self.use_multiple_gpu = True if not self.eval_model and ('gpu_ids' in args.keys() and len(args['gpu_ids']) > 1) else False
        self.determine_alpha_loss_func()
        self._init_laploss()
        self._init_percept_loss()
        if model_file is not None:
            self.load(model_file)

    def train(self, batch_item, keep_predictions=True):
        x = batch_item['image'].to(self.args['device'])
        self._iters += 1
        self.model.train()
        self.alpha_probe.train()
        z = self.model.encode(x)
        out = self.model.decode(z)
        if self.use_multiple_gpu:
            out = out.to('cuda:1')
            x = x.to('cuda:1')

        loss_ae = self.get_loss(x, out, is_test=False)['loss_ae']
        slice_between = batch_item['slice_between'].to(z.device)
        s_between_mix, pred_alphas, z_mix = self.synthesize_batch_images(batch_item, z)
        loss_ae_extra = self.get_extra_loss(slice_between, s_between_mix, z_mix, z=z,
                                               mask=None if not self.args['get_masks'] else batch_item['loss_mask'],
                                               is_test=False)
        loss_ae = loss_ae + loss_ae_extra

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


class AlphaTrainerEndToEnd(AlphaTrainer):

    def get_latent_loss(self, img_synthesized, img_reference, z_synthesized, device=None, is_test=False, z=None):
        # img_synthesized: decoding of the predicted latent mix for the slice-in-between
        # img_reference: the corresponding reference (original) for the img_synthesized (in-between slice)
        # we detach z_reference (slice in between encoding), because we don't want the loss caused by
        # alpha prediction to be accounted for by the encoder
        if is_test:
            self.model.eval()
            self.alpha_probe.eval()
        # difference to AlphaTrainer which uses torch.no_grad() here
        z_reference = self.model.encode(img_reference)
        z_loss = F.mse_loss(z_reference, z_synthesized)
        if z is not None:
            z_05 = self._get_mixup_latent(z)
            z_loss_05 = F.mse_loss(z_reference, z_05)
        z_mix_pred = self.model.encode(img_synthesized)
        z_mix_loss = F.mse_loss(z_reference, z_mix_pred)
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

    def synthesize_batch_images(self, batch_item, z, is_eval=False):

        feature_map = self.create_add_features(batch_item, z.size(0) // 2, z.device)
        z1, z3 = z.split(z.size(0) // 2, dim=0)
        z_reshape = torch.cat([z1, z3], dim=1)
        if is_eval:
            self.alpha_probe.eval()
            self.model.eval()
        # different to AlphaTrainer which uses z_reshape.detach()
        alpha = self.alpha_probe(z_reshape, feature_map)
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

        # difference to AlphaTrainer which uses with torch.no_grad():
        s_between_mix = self.model.decode(z_mix)
        if is_eval:
            self.alpha_probe.train()
            self.model.train()
        return s_between_mix, alpha, z_mix