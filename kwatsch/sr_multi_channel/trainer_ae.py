import os
import torch
import numpy as np
import torch.nn.functional as F
from kwatsch.trainer_ae import AEBaseTrainer
from kwatsch.dice_loss import DiceLoss
from kwatsch.acai_utils import generate_recon_grid
from kwatsch.training_utils import save_image_grid
from torchvision.utils import make_grid


def generate_recon_grid(img_ref, label_ref, img_recons, pred_labels, nclasses=3, max_items=16):
    if img_ref.is_cuda:
        img_ref = img_ref.detach().cpu()
    if label_ref.is_cuda:
        label_ref = label_ref.detach().cpu()
    if img_recons.is_cuda:
        img_recons = img_recons.detach().cpu()
    if pred_labels.is_cuda:
        pred_labels = pred_labels.detach().cpu()
    # assuming that batch size of x is divisible by 4
    if img_recons.size(0) < max_items:
        max_items = img_recons.size(0)
    diff_img = img_ref[:max_items] - img_recons[:max_items]
    diff_labels = label_ref[:max_items] - pred_labels[:max_items]
    recons_grid = torch.cat([img_ref[:max_items], img_recons[:max_items], diff_img], dim=0)
    recons_grid = make_grid(recons_grid.detach().cpu(), max_items, padding=2, normalize=False, pad_value=0.5).numpy()
    # make sure we divide integer values for class labels e.g. {0, 1, 2, 3} because when saving we multiply by 255
    # then all values will be max-ed out
    label_grid = torch.cat([label_ref[:max_items] / nclasses, pred_labels[:max_items] / nclasses, diff_labels / nclasses], dim=0)
    label_grid = make_grid(label_grid.detach().cpu(), max_items, padding=2, normalize=False, pad_value=0.5).numpy()

    return recons_grid, label_grid


def generate_batch_compare_grid(batch_dict, s_mix_inbetween, label_mix_inbetween, recon_images, pred_labels, nclasses=3):
    # s_mix_between: synthesized image
    # label_mix_inbetween: synthesized labels
    # recon_images: reconstructed images
    # pred_labels: reconstructed labels
    s_inbetween = batch_dict['slice_between'][:, 0, None]
    img1, img3 = batch_dict['image'].split(batch_dict['image'].size(0) // 2, dim=0)
    img1, img3 = img1[:, 0, None], img3[:, 0, None]
    recon1, recon3 = recon_images.split(recon_images.size(0) // 2, dim=0)

    diff = s_inbetween - s_mix_inbetween
    img_grid = torch.cat([img1, recon1, s_inbetween,
                          s_mix_inbetween, diff, recon3, img3], dim=0)
    img_grid = make_grid(img_grid.detach().cpu(), s_inbetween.size(0), padding=2,
                         normalize=False, pad_value=0.5).numpy()
    # labels
    lbl_inbetween = batch_dict['slice_between'][:, 1, None]  # reference labels
    lbl1, lbl3 = batch_dict['image'].split(batch_dict['image'].size(0) // 2, dim=0)
    lbl1, lbl3 = lbl1[:, 1, None], lbl3[:, 1, None]
    pred_labels1, pred_labels3 = pred_labels.split(pred_labels.size(0) // 2, dim=0)
    diff = lbl_inbetween - label_mix_inbetween
    label_grid = torch.cat([lbl1 / nclasses, pred_labels1 / nclasses, lbl_inbetween / nclasses,
                           label_mix_inbetween / nclasses, diff / nclasses, pred_labels3 / nclasses,
                            lbl3 / nclasses], dim=0)
    label_grid = make_grid(label_grid.detach().cpu(), s_inbetween.size(0), padding=2,
                           normalize=False, pad_value=0.5).numpy()

    return img_grid, label_grid


class MultiChannelBaseTrainer(AEBaseTrainer):

    def _init_mask_loss(self, n_classes):
        if "gpu_ids" in self.args.keys():
            if self.use_multiple_gpu:
                gpu_id = self.args['gpu_ids'][1]
            else:
                gpu_id = 0
        else:
            gpu_id = 0
        self.mask_loss = DiceLoss(n_classes).to("cuda:{}".format(gpu_id))

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
            out['image'] = out['image'].to('cuda:1')
            out['soft_probs'] = out['soft_probs'].to('cuda:1')
            x = x.to('cuda:1')

        loss_ae = self.get_loss(x[:, 0, None], out['image'], is_test=False)['loss_ae']
        loss_labels = 0.1 * self.mask_loss(out['soft_probs'], x[:, 1].to(torch.int64))
        loss_total = loss_ae + loss_labels
        return_dict = self.get_latent_loss(reference=batch_item['slice_between'].to(z.device),
                                           alpha_from=self.alpha05[0],
                                           alpha_to=self.alpha05[0],
                                           z=z, no_grad=True)
        self.opt_ae.zero_grad()
        if not eval_mode:
            loss_total.backward()
            self.opt_ae.step()

        # if lr_scheduler is not None take a scheduling step
        if self.opt_sched_ae is not None:
            self.opt_sched_ae.step()

        self.losses['loss_ae'].append(loss_ae.item())
        self.losses['loss_label'].append(loss_labels.item())
        self.losses['loss_latent_1'].append(return_dict['loss_latent'].item())
        mix_result_dict = self._get_mixup_image(z=z, alpha_from=self.alpha05[0],
                                                alpha_to=self.alpha05[0], is_test=True)
        slice_inbetween_mix = mix_result_dict['slice_inbetween_mix'].detach().cpu()
        label_inbetween_mix = mix_result_dict['label_inbetween_mix'].detach().cpu()
        if keep_predictions:
            _, pred_labels = torch.max(out['soft_probs'], dim=1, keepdim=True)
            self.train_predictions = {'z_mix': return_dict['z_mix'].detach().cpu(),
                                      'pred_alphas': self.alpha05[0],
                                      'slice_inbetween_mix': slice_inbetween_mix,
                                      'label_inbetween_mix': label_inbetween_mix,
                                      'slice_inbetween_05': slice_inbetween_mix,
                                      "reconstruction": out['image'].detach().cpu(),
                                      "pred_labels": pred_labels.detach().cpu()}

    def validate(self, validation_batch, image_dict=None, frame_id=0, generate_images=True):
        # first: grid to visualize with matplotlib or tensorboard
        # second: reconstructed images from batch
        self.model.eval()
        img_grid_recons = None
        if validation_batch['image'].shape[0] > 16:
            return_dict = self._validate_chunks(validation_batch, chunk_size=16)
            z = return_dict['z']
        else:
            z = self.encode(validation_batch['image'], eval=True)
            return_dict = self.decode(z, eval=True)
        img_recons, soft_probs, pred_labels = return_dict['image'], return_dict['soft_probs'], return_dict['pred_labels']
        loss = self.get_loss(validation_batch['image'][:, 0, None], img_recons.detach(), is_test=True)['loss_ae']
        loss_labels = self.mask_loss(soft_probs.detach(), validation_batch['image'][:, 1].to(torch.int64))
        alpha_from = self.alpha05[0] if 'alpha_from' not in validation_batch.keys() else validation_batch['alpha_from']
        alpha_to = self.alpha05[0] if 'alpha_to' not in validation_batch.keys() else validation_batch['alpha_to']
        latent_loss_dict = self.get_latent_loss(reference=validation_batch['slice_between'].to(z.device), z=z,
                                                alpha_from=alpha_from,
                                                alpha_to=alpha_to,
                                                no_grad=True)
        img_recons = img_recons.detach().cpu()
        self.test_predictions = {'z': z.detach().cpu(), 'img_recons': img_recons,
                                 'z_device': z.device, 'pred_labels': pred_labels}
        if generate_images:
            img_grid_recons, pred_label_grid = generate_recon_grid(validation_batch['image'][:, 0, None],
                                                                   validation_batch['image'][:, 1, None],
                                                                    img_recons, pred_labels)
        self.losses_test['loss_ae'].append(loss.item())
        self.losses_test['loss_label'].append(loss_labels.item())
        self.losses_test['loss_latent_1'].append(latent_loss_dict['loss_latent'].item())
        self.save_best_val_model()
        if image_dict is not None:
            synthesized_vols, alphas = self._generate_val_volumes(image_dict, frame_id=frame_id)
            return {"img_grid_recons": img_grid_recons, "synthesized_vols": synthesized_vols,
                    "pred_label_grid": pred_label_grid,
                    "loss_ae": self.losses_test['loss_ae'][-1], 'alphas': alphas}
        # if image_dict is None return reconstructions only
        return {"img_grid_recons": img_grid_recons, "loss_ae": self.losses_test['loss_ae'][-1],
                "pred_label_grid": pred_label_grid}

    def _validate_chunks(self, validation_batch, chunk_size=8):
        b = validation_batch['image'].shape[0]
        if b % chunk_size == 0:
            chunks = b // chunk_size
        else:
            chunks = (b // chunk_size) + 1
        # print("BaseTrainer - _validate_chunks {} and {}".format(b, chunks))
        img_recons, soft_probs, z = None, None, None
        losses = {'loss_ae': 0, 'loss_ae_dist': 0}
        for x_chunk in torch.chunk(validation_batch['image'], chunks=chunks, dim=0):
            with torch.no_grad():
                z_x = self.model.encode(x_chunk)
                out_x = self.model.decode(z_x)
            loss_dict = self.get_loss(x_chunk[:, 0, None], out_x['image'], is_test=True, store_loss=False)
            losses['loss_ae'] += loss_dict['loss_ae']
            losses['loss_ae_dist'] += loss_dict['loss_ae_dist']
            img_recons = torch.cat([img_recons, out_x['image']], dim=0) if img_recons is not None else out_x['image']
            soft_probs = torch.cat([soft_probs, out_x['soft_probs']], dim=0) if soft_probs is not None else out_x['soft_probs']
            z = torch.cat([z, z_x], dim=0) if z is not None else z_x

        losses['loss_ae'] = losses['loss_ae'].item() * 1/chunks
        losses['loss_ae_dist'] = losses['loss_ae_dist'].item() * 1/chunks
        self.losses_test['loss_ae'].append(losses['loss_ae'])
        self.losses_test['loss_ae_dist'].append(losses['loss_ae_dist'])
        _, pred_labels = torch.max(soft_probs, dim=1, keepdim=True)
        return {'image': img_recons, 'soft_probs': soft_probs, 'z': z, 'pred_labels': pred_labels}

    def _get_mixup_image(self, **kwargs):
        z, is_test = kwargs.get('z'), kwargs.get('is_test', False)
        z_05 = self._get_mixup_latent(**kwargs)
        if is_test:
            with torch.no_grad():
                result_05 = self.decode(z_05, eval=is_test)
        else:
            result_05 = self.decode(z_05, eval=is_test)
        return {'z_mix': z_05, 'slice_inbetween_mix': result_05['image'], 'soft_prob_mix': result_05['soft_probs'],
                'label_inbetween_mix': result_05['pred_labels']}

    def predict(self, x, eval=True, chunk_size=16, clear_cache=False, **kwargs):
        z = self.encode(x, eval=eval, chunk_size=chunk_size, clear_cache=clear_cache, **kwargs)
        result_dict = self.decode(z, eval=eval, clear_cache=clear_cache, chunk_size=chunk_size, **kwargs)
        return result_dict

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
            # z = z.detach().cpu()
            # we need to chunk
            if b % chunk_size == 0:
                chunks = b // chunk_size
            else:
                chunks = (b // chunk_size) + 1
            img, soft_probs, pred_labels = None, None, None
            for z_chunk in torch.chunk(z, chunks=chunks, dim=0):
                # z_chunk = z_chunk.to('cuda')
                if eval:
                    with torch.no_grad():
                        out_x = model.decode(z_chunk)
                else:
                    out_x = model.decode(z_chunk)
                img = torch.cat([img, out_x['image']], dim=0) if img is not None else out_x['image']
                soft_probs = torch.cat([soft_probs, out_x['soft_probs']], dim=0) if soft_probs is not None else out_x['soft_probs']
                _, pred_labels_chunk = torch.max(out_x['soft_probs'], dim=1, keepdim=True)
                pred_labels = torch.cat([pred_labels, pred_labels_chunk], dim=0) if pred_labels is not None else pred_labels_chunk
            self.do_chunk = False
            return {'image': img, 'soft_probs': soft_probs, 'pred_labels': pred_labels}
        if not z.is_cuda:
            z = z.to('cuda')
        if eval:
            with torch.no_grad():
                out = model.decode(z)
        else:
            out = model.decode(z)
        _, out['pred_labels'] = torch.max(out['soft_probs'], dim=1, keepdim=True)

        return out

    def generate_train_images(self, **kwargs):
        epoch = kwargs.get('epoch')
        batch_item = kwargs.get('batch_item')

        train_grid_compare, pred_label_grid = generate_batch_compare_grid(batch_item,
                                                         self.train_predictions['slice_inbetween_mix'],
                                                         self.train_predictions['label_inbetween_mix'],
                                                         self.train_predictions['reconstruction'],
                                                         self.train_predictions['pred_labels'])
        fname = os.path.join(self.args['dir_images'], 'train_image_e{:03d}_{}.png'.format(epoch,
                                                                                          self.iters))
        save_image_grid(train_grid_compare * 255, filename=fname)
        save_image_grid(pred_label_grid * 255, filename=fname.replace("_image_", "_pred_labels_"))
        self.add_image_tensorboard(train_grid_compare, "synthesized/train")

    def end_epoch_processing(self, **kwargs):
        super().end_epoch_processing(**kwargs)
        epoch = kwargs.get('epoch')
        val_result_dict = kwargs.get('val_result_dict')

        fname = os.path.join(self.args['dir_images'], 'val_pred_labels_e{:03d}.png'.format(epoch))
        save_image_grid(val_result_dict['pred_label_grid'] * 255, filename=fname)
        # remember, self.epoch is initialized with 0 in trainer_ae.py
        self.epoch += 1


class MultiChannelTrainer(MultiChannelBaseTrainer):

    def __init__(self, args, ae, max_grad_norm=0, model_file=None, eval_mode=False, **kwargs):
        super(MultiChannelTrainer, self).__init__(args, ae, max_grad_norm, model_file, eval_mode, **kwargs)
        self._init_mask_loss(args['nclasses'])


"""
    ********************** MultiChannelCAISRTrainer **********************
"""


class MultiChannelCAISRTrainer(MultiChannelBaseTrainer):

    def __init__(self, args, ae, max_grad_norm=0, model_file=None, eval_mode=False, **kwargs):
        super(MultiChannelCAISRTrainer, self).__init__(args, ae, max_grad_norm, model_file, eval_mode, **kwargs)
        self._init_mask_loss(args['nclasses'])

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
            out['image'] = out['image'].to('cuda:1')
            out['soft_probs'] = out['soft_probs'].to('cuda:1')
            x = x.to('cuda:1')

        loss_ae = self.get_loss(x[:, 0, None], out['image'], is_test=False)['loss_ae']
        loss_labels = 0.1 * self.mask_loss(out['soft_probs'], x[:, 1].to(torch.int64))
        return_dict = self.get_latent_loss(reference=batch_item['slice_between'].to(z.device),
                                           alpha_from=self.alpha05[0],
                                           alpha_to=self.alpha05[0],
                                           z=z, no_grad=True)
        mix_result_dict = self._get_mixup_image(z=z, alpha_from=self.alpha05[0],
                                                alpha_to=self.alpha05[0], is_test=False)
        slice_between = batch_item['slice_between'].to(z.device)
        loss_ae_extra = self.get_extra_loss(slice_between, mix_result_dict['slice_inbetween_mix'],
                                            mix_result_dict['soft_prob_mix'],
                                            is_test=False)
        loss_total = loss_ae + loss_ae_extra + loss_labels

        self.opt_ae.zero_grad()
        if not eval_mode:
            loss_total.backward()
            self.opt_ae.step()

        # if lr_scheduler is not None take a scheduling step
        if self.opt_sched_ae is not None:
            self.opt_sched_ae.step()

        self.losses['loss_ae'].append(loss_ae.item())
        self.losses['loss_label'].append(loss_labels.item())
        self.losses['loss_latent_1'].append(return_dict['loss_latent'].item())

        slice_inbetween_mix = mix_result_dict['slice_inbetween_mix'].detach().cpu()
        label_inbetween_mix = mix_result_dict['label_inbetween_mix'].detach().cpu()
        if keep_predictions:
            _, pred_labels = torch.max(out['soft_probs'], dim=1, keepdim=True)
            self.train_predictions = {'z_mix': return_dict['z_mix'].detach().cpu(),
                                      'pred_alphas': self.alpha05[0],
                                      'slice_inbetween_mix': slice_inbetween_mix,
                                      'label_inbetween_mix': label_inbetween_mix,
                                      'slice_inbetween_05': slice_inbetween_mix,
                                      "reconstruction": out['image'].detach().cpu(),
                                      "pred_labels": pred_labels.detach().cpu()}

    def validate(self, validation_batch, image_dict=None, frame_id=0, generate_images=True):
        val_result_dict = super().validate(validation_batch, image_dict=image_dict, frame_id=frame_id,
                                           generate_images=generate_images)

        z = self.test_predictions['z'].to(self.test_predictions['z_device'])
        slice_between = validation_batch['slice_between'].to(z.device)
        return_dict_sbi = self._get_mixup_image(z=z, alpha_from=self.alpha05[0],
                                                alpha_to=self.alpha05[0], is_test=True)
        _ = self.get_extra_loss(slice_between, return_dict_sbi['slice_inbetween_mix'], return_dict_sbi['soft_prob_mix'],
                                is_test=True)
        self.save_best_val_model()
        return val_result_dict

    def get_extra_loss(self, slice_between, s_between_mix, soft_probs_between_mix, is_test=False):
        if self.args['use_loss_annealing']:
            # print("Loss weight {:.6f}".format(self.loss_weights[self.epoch]))
            loss_extra_image, loss_extra_labels = self.get_extra_image_loss(slice_between, s_between_mix,
                                                                            soft_probs_between_mix)
            loss_extra_image = self.loss_weights[self.epoch] * loss_extra_image
            loss_extra_labels = self.loss_weights[self.epoch] * loss_extra_labels
        else:
            loss_extra_image, loss_extra_labels = self.get_extra_image_loss(slice_between, s_between_mix,
                                                                            soft_probs_between_mix)
            loss_extra_image = self.args['ex_loss_weight1'] * loss_extra_image
            loss_extra_labels = self.args['ex_loss_weight1'] * loss_extra_labels
        loss_ae_extra = loss_extra_image + loss_extra_labels

        if is_test:
            self.losses_test['loss_ae_extra'].append(loss_ae_extra.item())
            self.losses_test['loss_ae_dist_extra'].append(loss_extra_image.item())
            self.losses_test['loss_ae_dist_labels'].append(loss_extra_labels.item())
        else:
            self.losses['loss_ae_extra'].append(loss_ae_extra.item())
            self.losses['loss_ae_dist_extra'].append(loss_extra_image.item())
            self.losses['loss_ae_dist_labels'].append(loss_extra_labels.item())
        return loss_ae_extra

    def get_extra_image_loss(self, reference, synthesized, soft_probs_synthesized):
        """
        Args:
            reference: is tensor with shape [batch, 2, y, x]. dim1: index0=image, index1=labels
            synthesized: synthesized image [batch, 1, y, x]
            soft_probs_synthesized: synthesized labels as softmax probs [batch, nclasses, y, x]

        Returns:

        """
        if self.percept_criterion is not None and self.image_mix_loss_func == 'perceptual':
            if self.use_multiple_gpu:
                reference, synthesized = reference.to('cuda:1'), synthesized.to('cuda:1')
            loss_image = self.percept_criterion(reference[:, 0, None], synthesized, normalize=True).mean()
        else:
            if self.use_multiple_gpu:
                reference, synthesized = reference.to('cuda:1'), synthesized.to('cuda:1')
            loss_image = F.mse_loss(reference[:, 0, None], synthesized)
            # loss_image = 1 - self.ssim_criterion(synthesized, reference)
            if self.laploss is not None:
                loss_image = self.laploss(synthesized, reference[:, 0, None]) + loss_image

            if self.use_multiple_gpu:
                loss_image = loss_image.to('cuda:1')
        loss_labels = self.mask_loss(soft_probs_synthesized, reference[:, 1].to(torch.int64))

        return loss_image, loss_labels

    def save_best_val_model(self, **kwargs):
        super().save_best_val_model()
        if len(self.mean_losses_test['loss_ae_extra']) > 1:
            if np.argmin(self.mean_losses_test['loss_ae_extra']) + 1 == len(self.mean_losses_test['loss_ae_extra']):
                # the last loss is the lowest, save model
                # although slightly awkward (otherwise have to adjust too much other stuff,
                # we pass self.epoch which is initialized by 0, therefore plus 1
                fname = os.path.join(self.args['dir_models'], 'caisr.models')
                self.save_models(fname, self.epoch + 1)