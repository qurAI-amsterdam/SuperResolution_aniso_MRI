import os
import torch
import numpy as np
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
from kwatsch.acai_utils import generate_recon_grid
import torch.nn.functional as F
from kwatsch.lap_pyramid_loss import LapLoss
from lpips.perceptual import PerceptualLoss
from evaluate.evaluate_image import evaluate_image, create_compare_image
from collections import defaultdict
from kwatsch.training_utils import save_image_grid
from kwatsch.training_utils import generate_batch_compare_grid


class BaseTrainer(object):

    def _init_scheduler(self):
        if "use_lr_scheduler" in self.args.keys() and self.args['use_lr_scheduler']:
            print("WARNING - {} - usig lr_scheduling - {}".format(self.__class__.__name__, self.args['lr_iter_max']))
            self.opt_sched_ae = optim.lr_scheduler.CosineAnnealingLR(self.opt_ae, self.args["lr_iter_max"],
                                                                     eta_min=0, last_epoch=-1)

    def _init_percept_loss(self):
        if self.args['use_percept_loss']:
            self.ae_loss_func = "perceptual"
        else:
            self.ae_loss_func = "mse"
        if not self.eval_model and (('use_percept_loss' in self.args.keys() and self.args['use_percept_loss']) or \
                                    ('image_mix_loss_func' in self.args.keys() and self.args[
                                        'image_mix_loss_func'] == "perceptual") or\
                                    ('alpha_loss_func' in self.args.keys() and self.args['alpha_loss_func'] == "perceptual")):
            if "gpu_ids" in self.args.keys():
                if self.use_multiple_gpu:
                    gpu_id = self.args['gpu_ids'][1]
                else:
                    gpu_id = 0
            else:
                gpu_id = 0
            print("Warning - Base Trainer - LPIPS on GPU-ID {}".format(gpu_id))
            self.percept_criterion = PerceptualLoss(
                model='net-lin', net='vgg', use_gpu=self.args['device'].startswith('cuda'), gpu_ids=[int(gpu_id)]
            )
        else:
            self.percept_criterion = None

    def _init_laploss(self):
        if not self.eval_model and 'use_laploss' in self.args.keys() and self.args['use_laploss']:
            if self.use_multiple_gpu:
                l_device = 'cuda:1'
            else:
                l_device = "cuda:0"

            self.laploss = LapLoss(channels=1, device=l_device)
        else:
            self.laploss = None

    def determine_image_mix_loss_func(self):
        if "image_mix_loss_func" in self.args.keys():
            self.image_mix_loss_func = self.args['image_mix_loss_func']
        else:
            if self.args['use_percept_loss']:
                self.image_mix_loss_func = "perceptual"
            else:
                self.image_mix_loss_func = "mse"

    def validate(self, validation_batch, image_dict=None, frame_id=8, generate_images=True):
        # first: grid to visualize with matplotlib or tensorboard
        # second: reconstructed images from batch
        self.model.eval()
        img_grid_recons = None
        if validation_batch['image'].shape[0] > 16:
            img_recons, z = self._validate_chunks(validation_batch, chunk_size=16)
        else:
            z = self.encode(validation_batch['image'], eval=True)
            img_recons = self.decode(z, eval=True)
        loss = self.get_loss(validation_batch['image'], img_recons, is_test=True)['loss_ae']
        alpha_from = self.alpha05[0] if 'alpha_from' not in validation_batch.keys() else validation_batch['alpha_from']
        alpha_to = self.alpha05[0] if 'alpha_to' not in validation_batch.keys() else validation_batch['alpha_to']
        latent_loss_dict = self.get_latent_loss(reference=validation_batch['slice_between'].to(z.device), z=z,
                                                alpha_from=alpha_from,
                                                alpha_to=alpha_to,
                                                no_grad=True)
        img_recons = img_recons.detach().cpu()
        self.test_predictions = {'z': z.detach().cpu(), 'img_recons': img_recons,
                                 'z_device': z.device}
        if generate_images:
            img_grid_recons = generate_recon_grid(validation_batch['image'], img_recons)
        self.losses_test['loss_ae'].append(loss.item())
        self.losses_test['loss_latent_1'].append(latent_loss_dict['loss_latent'].item())
        if self.epoch > self.args['epoch_threshold'] and 'vae' not in self.args['model']:
            # print("INFO - BaseTrainer - saved models e{} > {}".format(self.epoch, self.args['epoch_threshold']))
            self.save_best_val_model()
        if image_dict is not None:
            synthesized_vols, alphas = self._generate_val_volumes(image_dict, frame_id=frame_id)
            return {"img_grid_recons": img_grid_recons, "synthesized_vols": synthesized_vols,
                    "loss_ae": self.losses_test['loss_ae'][-1], 'alphas': alphas}
        # if image_dict is None return reconstructions only
        return {"img_grid_recons": img_grid_recons, "loss_ae": self.losses_test['loss_ae'][-1]}

    def save_best_val_model(self, **kwargs):
        if len(self.mean_losses_test['loss_ae_dist']) > 1:
            if np.argmin(self.mean_losses_test['loss_ae_dist']) + 1 == len(self.mean_losses_test['loss_ae_dist']):
                # the last loss is the lowest, save model
                # although slightly awkward (otherwise have to adjust too much other stuff,
                # we pass self.epoch which is initialized by 0, therefore plus 1
                fname = os.path.join(self.args['dir_models'], 'ae.models')
                self.save_models(fname, self.epoch + 1)

    def _validate_chunks(self, validation_batch, chunk_size=8):
        b = validation_batch['image'].shape[0]
        if b % chunk_size == 0:
            chunks = b // chunk_size
        else:
            chunks = (b // chunk_size) + 1
        # print("BaseTrainer - _validate_chunks {} and {}".format(b, chunks))
        img_recons, z = None, None
        losses = {'loss_ae': 0, 'loss_ae_dist': 0}
        for x_chunk in torch.chunk(validation_batch['image'], chunks=chunks, dim=0):
            with torch.no_grad():
                z_x = self.model.encode(x_chunk)
                out_x = self.model.decode(z_x)
            loss_dict = self.get_loss(x_chunk, out_x, is_test=True, store_loss=False)
            losses['loss_ae'] += loss_dict['loss_ae']
            losses['loss_ae_dist'] += loss_dict['loss_ae_dist']
            img_recons = torch.cat([img_recons, out_x], dim=0) if img_recons is not None else out_x
            z = torch.cat([z, z_x], dim=0) if z is not None else z_x

        losses['loss_ae'] = losses['loss_ae'].item() * 1/chunks
        losses['loss_ae_dist'] = losses['loss_ae_dist'].item() * 1/chunks
        self.losses_test['loss_ae'].append(losses['loss_ae'])
        self.losses_test['loss_ae_dist'].append(losses['loss_ae_dist'])

        return img_recons, z

    def _chunk_perceptual_loss(self, image, reference, chunk_size=16):
        b = image.shape[0]
        if b % chunk_size == 0:
            chunks = b // chunk_size
        else:
            chunks = (b // chunk_size) + 1

        losses = 0
        for img_chunk, ref_chunk in zip(torch.chunk(image, chunks=chunks, dim=0), torch.chunk(reference, chunks=chunks, dim=0)):
            with torch.no_grad():
                losses += self.percept_criterion(img_chunk, ref_chunk, normalize=True).mean()
        return losses * 1/chunks

    def _generate_val_volumes(self, image_dict, frame_id):
        synthesized_vols, alphas = defaultdict(dict), defaultdict(dict)
        eval_patch_size = self.args['width'] if 'eval_patch_size' not in self.args.keys() else self.args[
            'eval_patch_size']
        # loop over dictionary key integer patient ID. Contains again dict with key 'image' etc. 4d volumes
        for p_id in image_dict.keys():
            # return dict with three keys (see below). Each of these three has frame_id as dict key
            return_dict = evaluate_image(self, image_dict[p_id], frame_id=frame_id, downsample_steps=2,
                                         eval_patch_size=eval_patch_size)
            org_i, synth_i = return_dict["orig_images"][frame_id], return_dict["synth_images"][frame_id]
            img_grid_compare = create_compare_image(org_i, synth_i)
            synthesized_vols[p_id], alphas[p_id] = img_grid_compare, return_dict["pred_alphas"][frame_id]

        return synthesized_vols, alphas

    def get_loss(self, reference, recons, is_test=False, store_loss=True):
        if self.ae_loss_func == "perceptual" and self.percept_criterion is not None:
            if is_test:
                if reference.shape[0] > 16:
                    loss_ae_dist = self._chunk_perceptual_loss(recons, reference)
                else:
                    with torch.no_grad():
                        loss_ae_dist = self.percept_criterion(recons, reference, normalize=True).mean()
            else:
                loss_ae_dist = self.percept_criterion(recons, reference, normalize=True).mean()
            if self.use_multiple_gpu:
                loss_ae_dist = loss_ae_dist.to(device=reference.device)
        else:
            loss_ae_dist = F.mse_loss(recons, reference, reduction='mean')
            # loss_ae_dist = F.smooth_l1_loss(recons, reference, reduction='mean')
            if self.ssim_criterion is not None:
                ssim_loss = 0.135 * (1 - self.ssim_criterion(recons, reference))
                loss_ae_dist = loss_ae_dist + ssim_loss

        if self.laploss is not None:
            loss_laplacian = self.laploss(recons, reference)
        else:
            loss_laplacian = 0
        if store_loss:
            if is_test:
                self.losses_test['loss_ae_dist'].append(loss_ae_dist.item())
                if self.laploss is not None:
                    self.losses_test['loss_laploss'].append(loss_laplacian.item())
            else:
                self.losses['loss_ae_dist'].append(loss_ae_dist.item())
                if self.laploss is not None:
                    self.losses['loss_laploss'].append(loss_laplacian.item())

        return {"loss_ae": loss_ae_dist + loss_laplacian, 'loss_ae_dist': loss_ae_dist,
                "loss_laploss": loss_laplacian}

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

    @property
    def iters(self):
        return self._iters

    def predict(self, x, eval=True, chunk_size=16, clear_cache=False, **kwargs):
        if eval:
            self.model.eval()
        else:
            self.model.train()
        b, c, w, h = x.shape
        if w == 256 and h == 256 and b > chunk_size:
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
                        out_x = self.model(x_chunk)
                else:
                    out_x = self.model(x_chunk)
                out = torch.cat([out, out_x.detach().cpu()], dim=0) if out is not None else out_x.detach().cpu()
            return out
        if eval:
            with torch.no_grad():
                out = self.model(x)
        else:
            out = self.model(x)
        return out

    def encode(self, x, eval=True, clear_cache=False, chunk_size=16, **kwargs):
        model = self._use_sr_model(kwargs.get("use_sr_model", False), origin="*encode*")
        # print("BaseTrainer - encode - model_sr", kwargs.get("use_sr_model", False), False if self.model_sr is None else True)
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
                out = torch.cat([out, out_x.detach().cpu()], dim=0) if out is not None else out_x.detach().cpu()
            return out
        if eval:
            with torch.no_grad():
                out = model.encode(x)
        else:
            out = model.encode(x)
        return out

    def decode(self, z, eval=True, clear_cache=False, chunk_size=16, **kwargs):
        model = self._use_sr_model(kwargs.get("use_sr_model", False), origin="*decode*")
        if eval:
            model.eval()
            if clear_cache:
                torch.cuda.empty_cache()
        else:
            model.train()
        b, c, w, h = z.shape
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

    def _use_sr_model(self, use_sr_model=False, **kwargs):
        origin = kwargs.get('origin', 'None')
        if not use_sr_model:
            model = self.model
            # print("Trainer - ae - {}".format(origin))
        elif self.model_sr is not None:
            model = self.model_sr
            # print("Trainer - caisr - {}".format(origin))
        else:
            # print("Trainer - ae - {}".format(origin))
            model = self.model
        return model

    def _get_mixup_image(self, **kwargs):
        z, is_test = kwargs.get('z'), kwargs.get('is_test', False)
        z_05 = self._get_mixup_latent(**kwargs)
        if is_test:
            with torch.no_grad():
                s_slice_05 = self.model.decode(z_05)
        else:
            s_slice_05 = self.model.decode(z_05)
        return {'z_mix': z_05, 'slice_inbetween_mix': s_slice_05}

    def _get_mixup_latent(self, **kwargs):
        z = kwargs.get('z')
        z_05 = 0.5 * z[:z.size(0) // 2] + 0.5 * z[z.size(0) // 2:]
        return z_05

    def save_models(self, fname, epoch):
        torch.save({'model_dict_ae': self.model.state_dict(),
                    'optimizer_dict_ae': self.opt_ae.state_dict(),
                    'epoch': epoch}, fname)

    def load(self, fname):
        state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict['model_dict_ae'])
        self.opt_ae.load_state_dict(state_dict['optimizer_dict_ae'])
        print("INFO - {} Loaded model parameters from {}".format(self.__class__.__name__, fname))

    def load_caisr(self, fname):
        state_dict = torch.load(fname)
        self.model_sr.load_state_dict(state_dict['model_dict_ae'])
        print("Warning - {} Loaded model parameters of extra SR model from {}".format(self.__class__.__name__, fname))

    def init_tensorboard(self, output_directory):
        self.tb_writer = SummaryWriter(log_dir=os.path.join(output_directory, "tb"), comment=str(self.args))

    def show_loss_on_tensorboard(self, eval_type='train'):
        if eval_type == "train":
            loss_dict = self.losses
            mean_losses = self.mean_losses
            self.loss_iters.append(self.iters)
        else:
            loss_dict = self.losses_test
            mean_losses = self.mean_losses_test
        for loss_key in loss_dict.keys():
            mean_value = np.mean(np.array(loss_dict[loss_key]))
            log_type = "{}/{}".format(loss_key, eval_type)
            if self.args['log_tensorboard']:
                self.tb_writer.add_scalar(log_type, mean_value, self.iters)
            mean_losses[loss_key].append(mean_value)

    def add_image_tensorboard(self, image, log_type):
        self.tb_writer.add_image(log_type, image, self.iters)

    def add_hist_tensorboard(self, array1d, log_type):
        if isinstance(array1d, list):
            array1d = np.array(array1d)
        self.tb_writer.add_histogram(log_type, array1d, self.iters)

    def generate_train_images(self, **kwargs):
        epoch = kwargs.get('epoch')
        batch_item = kwargs.get('batch_item')
        train_grid_compare = generate_batch_compare_grid(batch_item,
                                                         self.train_predictions['slice_inbetween_mix'],
                                                         self.train_predictions['reconstruction'])
        fname = os.path.join(self.args['dir_images'], 'train_image_e{:03d}_{}.png'.format(epoch,
                                                                                          self.iters))
        save_image_grid(train_grid_compare * 255, filename=fname)
        if self.args['log_tensorboard']:
            self.add_image_tensorboard(train_grid_compare, "synthesized/train")

    def end_epoch_processing(self, **kwargs):
        epoch = kwargs.get('epoch')
        val_result_dict = kwargs.get('val_result_dict')
        fname = os.path.join(self.args['dir_models'], '{:0d}.models'.format(epoch))
        if self.epoch > self.args['epoch_threshold']:
            self.save_models(fname, epoch)
        self.save_losses()
        # Save example validation volumes as .png to disc
        fname = os.path.join(self.args['dir_images'], 'val_image_e{:03d}_xxx.png'.format(epoch))
        if 'synthesized_vols' in val_result_dict.keys() and val_result_dict['synthesized_vols'] is not None:
            for p_id, s_img in val_result_dict['synthesized_vols'].items():
                save_image_grid(s_img * 255, filename=fname.replace('xxx', 'p{:03d}'.format(p_id)))
        fname = os.path.join(self.args['dir_images'], 'val_recons_e{:03d}.png'.format(epoch))
        save_image_grid(val_result_dict['img_grid_recons'] * 255, filename=fname)
        # remember, self.epoch is initialized with 0 in trainer_ae.py
        self.epoch += 1

    def save_model(self, **kwargs):
        epoch, with_iters = kwargs.get('epoch'), kwargs.get('with_iters', False)

        if not with_iters:
            fname = os.path.join(self.args['dir_models'], '{:0d}.models'.format(epoch))
        else:
            fname = os.path.join(self.args['dir_models'], '{:0d}_{}.models'.format(epoch, self.iters))
        self.save_models(fname, epoch)

    @staticmethod
    def load_losses(path_to_exper):
        path_to_exper = os.path.expanduser(path_to_exper)
        iters = np.load(os.path.join(path_to_exper, "loss_iters.npz"))['loss_iters']
        losses_train = np.load(os.path.join(path_to_exper, "losses_train.npz"))
        losses_test = np.load(os.path.join(path_to_exper, "losses_test.npz"))
        losses_train = {arr_key: losses_train[arr_key] for arr_key in losses_train.files}
        losses_test = {arr_key: losses_test[arr_key] for arr_key in losses_test.files}
        return iters, losses_train, losses_test

    def save_losses(self):
        save_file = os.path.join(self.args['output_dir'], "loss_iters.npz")
        np.savez(save_file, loss_iters=np.array(self.loss_iters))
        save_file = os.path.join(self.args['output_dir'], "losses_train.npz")
        np.savez(save_file, **self.mean_losses)
        save_file = os.path.join(self.args['output_dir'], "losses_test.npz")
        np.savez(save_file, **self.mean_losses_test)

    def init_weight_ramp(self, epochs):
        x = np.linspace(-2, 10, epochs)
        y = torch.sigmoid(torch.from_numpy(x)) * self.args.get('ex_loss_weight1', 0.001)
        self.loss_weights = y.numpy()

    def init_weight_annealing(self, epochs):
        x = np.linspace(-5, 5, epochs)
        y = torch.sigmoid(torch.from_numpy(x)) * self.args.get('ex_loss_weight1', 0.001)
        self.loss_weights = y.numpy()[::-1]

    def reset_losses(self):
        if len(self.losses) > 0:
            for key in self.losses.keys():
                self.losses[key] = []
        if len(self.losses_test) > 0:
            for key in self.losses_test.keys():
                self.losses_test[key] = []


import matplotlib.pyplot as plt


def show_loss_curves(iters, losses_train, losses_test, iter_range=None):
    if iter_range is None:
        iter_range = slice(0, len(iters), None)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots(3, 1)
    ax[0].plot(iters[iter_range], losses_train['loss_ae_dist'][iter_range], c='b', label='tr')
    ax[0].plot(iters[iter_range], losses_test['loss_ae_dist'][iter_range], c='r', label='te')
    ax[1].plot(iters[iter_range], losses_train['loss_ae_dist_extra'][iter_range], c='b', label='tr')
    ax[1].plot(iters[iter_range], losses_test['loss_ae_dist_extra'][iter_range], c='r', label='te')
    ax[2].plot(iters[iter_range], losses_train['loss_latent_1'][iter_range], c='b', label='tr')
    ax[2].plot(iters[iter_range], losses_test['loss_latent_1'][iter_range], c='r', label='te')
    ax[0].set_title("Reconstruction"), ax[1].set_title("Extra image loss"), ax[2].set_title("latent mse")
    ax[0].legend(loc="best"), ax[1].legend(loc="best"), ax[2].legend(loc="best")
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    plt.show()