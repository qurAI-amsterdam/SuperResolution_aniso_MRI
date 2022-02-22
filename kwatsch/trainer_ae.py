import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from collections import defaultdict
from kwatsch.lap_pyramid_loss import LapLoss
from lpips.perceptual import PerceptualLoss
from kwatsch.base_trainer import BaseTrainer
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
norm_layer = nn.BatchNorm2d


class AEBaseTrainer(BaseTrainer):

    def __init__(self, args, ae, max_grad_norm=0, model_file=None, eval_mode=False, **kwargs):
        super(AEBaseTrainer, self).__init__()
        self.args = args
        self.model = ae
        self.model_sr = kwargs.get('model_sr', None)
        self.eval_model = eval_mode
        self.model_file = model_file
        self.do_chunk = False
        # This is only important for models predicting alpha coefficients. Used in create_super_volume function
        self.eval_fixed_coeff = True
        momentum = 0.9 if 'momentum' not in args.keys() else args['momentum']
        self.opt_ae = optim.Adam(self.model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'],
                                 betas=(momentum, 0.999))
        self.opt_sched_ae = None
        self._init_scheduler()
        # these losses will be reset after logged on tensorboard (see method reset_losses)
        self.losses = defaultdict(list)
        self.losses_test = defaultdict(list)
        # these will be saved for analysis purposes (sometimes tensorboard losses are not practical)
        self.loss_iters = list()
        self.mean_losses = defaultdict(list)
        self.mean_losses_test = defaultdict(list)
        self.train_predictions, self.test_predictions = None, None
        self._iters = 1
        self.max_grad_norm = max_grad_norm
        self.use_multiple_gpu = True if not self.eval_model and (
                    'gpu_ids' in args.keys() and len(args['gpu_ids']) > 1) else False
        if self.eval_model:
            print("INFO - {} - model is in eval mode {}".format(self.__class__.__name__, self.eval_model))
        else:
            print("INFO - Initializing trainer {} - using multiple gpus {}".format(self.__class__.__name__,
                  self.use_multiple_gpu))
        # 23-12-2020 Disabled self.alpha05 because I think it's not used anymore
        self.alpha05 = torch.cuda.FloatTensor([0.5], device=torch.device('cuda:0'))[:, None, None, None]
        # ok this is clamsy. But in brain trainers we specified alpha_from & alpha_to, scalar values.
        self._init_laploss()
        self._init_percept_loss()
        self.determine_image_mix_loss_func()
        self.ssim_criterion = None
        # we use this for ramping up constraint loss
        self.epoch = 0
        # self.init_weight_ramp(self.args['epochs'])
        self.init_weight_annealing(self.args['epochs'])
        if 'use_ssim_loss' in self.args.keys() and self.args['use_ssim_loss']:
            raise NotImplementedError("ERROR - Disabled SSIM as loss when upgrading pytorch to 1.9 version!")
            # self.ssim_criterion = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        if model_file is not None:
            self.model_file = model_file
            self.load(model_file)
        if self.model_sr is not None and kwargs.get("model_file_sr", None) is not None:
            self.model_file_sr = kwargs.get("model_file_sr")
            self.load_caisr(self.model_file_sr)

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
                                           alpha_from=self.alpha05[0],
                                           alpha_to=self.alpha05[0],
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
        mix_result_dict = self._get_mixup_image(z=z, alpha_from=self.alpha05[0],
                                                alpha_to=self.alpha05[0], is_test=True)
        slice_inbetween_mix = mix_result_dict['slice_inbetween_mix'].detach().cpu()
        if keep_predictions:
            self.train_predictions = {'z_mix': return_dict['z_mix'].detach().cpu(),
                                      'pred_alphas': self.alpha05[0],
                                      'slice_inbetween_mix': slice_inbetween_mix,
                                      'slice_inbetween_05': slice_inbetween_mix,
                                      "reconstruction": out.detach().cpu()}

