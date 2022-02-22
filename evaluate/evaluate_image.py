import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid
from evaluate.common import create_super_volume
from datasets.shared_transforms import CenterCrop, AdjustToPatchSize
from evaluate.metrics import compute_ssim_for_batch, compute_psnr_for_batch
from evaluate.metrics import compute_vif_for_batch


def compute_stats(trainer, original_img, r_s_img, normalize=True,
                 downsample_steps=None, is_conv_meth=False):
    ssim_res = compute_ssim_for_batch(original_img, r_s_img, eval_axis=0,
                        downsample_steps=downsample_steps,
                                           conv_interpol=is_conv_meth, normalize=normalize)
    psnr_res = compute_psnr_for_batch(original_img, r_s_img, eval_axis=0,
                                           downsample_steps=downsample_steps,
                                           conv_interpol=is_conv_meth, normalize=normalize)
    vif_res = compute_vif_for_batch(original_img, r_s_img, eval_axis=0,
                                           downsample_steps=downsample_steps,
                                           conv_interpol=is_conv_meth, normalize=normalize)
    if isinstance(original_img, np.ndarray):
        original_img = torch.from_numpy(original_img)
    if isinstance(r_s_img, np.ndarray):
        r_s_img = torch.from_numpy(r_s_img)
    if downsample_steps is None:
        lpips_res = trainer.percept_criterion(original_img[:, None], r_s_img[:, None],
                                               normalize=True).mean().item()
    else:
        lpips_res = trainer.percept_criterion(
            original_img[:, None][::downsample_steps], r_s_img[:, None][::downsample_steps],
                                               normalize=True).mean().item()
    return ssim_res, psnr_res, vif_res, lpips_res


def evaluate_image(trainer, data_dict, frame_id=None, eval_patch_size=128,
                  downsample_steps=2, transform=None):
    """
        data_dict: generate e.g. with get_4d_image_array (from datasets.ACDC.data4d_simple)
    """
    if transform is None:
        transform = transforms.Compose([AdjustToPatchSize(tuple((eval_patch_size, eval_patch_size))),
                                        CenterCrop(eval_patch_size)])
    images4d_hr = data_dict['image']
    mask = data_dict['mask'] if "get_mask" in trainer.args.keys() and trainer.args['get_mask'] else None
    labels = None if 'labels' not in data_dict.keys() else data_dict['labels']
    patid = data_dict['patient_id']
    p_id = int(patid.replace("patient", ""))
    z_spacing = data_dict['spacing'][0]
    num_frames = images4d_hr.shape[0]
    if frame_id is None:
        f_range = np.arange(0, num_frames)
    else:
        if frame_id >= num_frames:
            frame_id = num_frames - 1
        f_range = np.arange(frame_id, frame_id+1)
    synth_images, orig_images, alphas = dict(), dict(), dict()
    for f_id in f_range:
        norm_frame_id = (f_id + 1) / num_frames
        images3d_hr = transform({'image': images4d_hr[f_id]})['image']
        images3d_hr = torch.from_numpy(images3d_hr)
        labels3d_hr = None if labels is None else torch.from_numpy(transform({'image': labels[f_id]})['image'])
        if mask is not None:
            mask = transform({'image': mask[f_id]})['image']
        num_slices = images3d_hr.shape[0]
        feature_dict = {'anatomy': 'cardiac', 'spacing': z_spacing,
                        'orig_num_slices': num_slices, 'norm_frame_id': norm_frame_id}

        create_super_volume_dict = create_super_volume(trainer,
                                                            images3d_hr, alpha_range=[0.5], use_original=False,
                                                            hierarchical=False, downsample_steps=downsample_steps,
                                                            generate_inbetween_slices=True, train_patch_size=None,
                                                            feature_dict=feature_dict, labels=labels3d_hr)
        new_images_alpha, pred_alphas = create_super_volume_dict['upsampled_image'], \
                                        create_super_volume_dict['pred_alphas']
        synth_images[f_id] = new_images_alpha.detach().cpu().squeeze().numpy()
        orig_images[f_id] = images3d_hr.detach().cpu().squeeze().numpy()
        alphas[f_id] = pred_alphas.squeeze()
    return {'orig_images': orig_images, 'synth_images': synth_images, 'pred_alphas': alphas}


def create_compare_image(real_img, synth_img, downsample_steps=2):
    """
        both numpy arrays of shape [#slices, y, x]
    """
    # get synthesized images from complete volume
    if real_img.shape[0] % downsample_steps == 0:
        num_slices = real_img.shape[0] - 1
        real_img, synth_img = real_img[:-1], synth_img[:-1]
    else:
        num_slices = real_img.shape[0]
    s_mask = np.ones(num_slices).astype(np.bool)
    r_mask = np.zeros(num_slices).astype(np.bool)
    s_mask[np.arange(0, num_slices)[::downsample_steps]] = False
    r_mask[np.arange(0, num_slices)[::downsample_steps]] = True
    slices1, slice3 = real_img[r_mask][:-1], real_img[r_mask][1:]
    slices1_recons, slices3_recons = synth_img[r_mask][:-1], synth_img[r_mask][1:]
    synth_img, real_img = synth_img[s_mask], real_img[s_mask]
    diff = real_img - synth_img
    # Changed this. Showing reconstructions as well as synthesized images
    img_grid = np.concatenate([slices1[:, None], slices1_recons[:, None], real_img[:, None], synth_img[:, None], diff[:, None],
                               slices3_recons[:, None], slice3[:, None]], axis=0)
    # img_grid = make_grid(torch.from_numpy(img_grid), slices1.shape[0], padding=2,
    #                      normalize=False, pad_value=0.5).numpy().squeeze().transpose(1, 2, 0)
    img_grid = make_grid(torch.from_numpy(img_grid), slices1.shape[0], padding=2,
                         normalize=False, pad_value=0.5).numpy()
    return img_grid