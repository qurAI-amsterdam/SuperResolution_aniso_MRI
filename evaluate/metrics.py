import sklearn
try:
    from skimage.measure import compare_ssim as ssim_metric
    from skimage.measure import compare_psnr as psnr_metric
except ImportError:
    from skimage.metrics import structural_similarity as ssim_metric
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import numpy as np
import torch
import copy
from math import log10, sqrt
from evaluate.vifvec import vifp_mscale  # this one is faster and computes roughly the same values
# from evaluate.vifvec_alternative import vifvec_alternative
from datasets.common import rescale_intensities
from lpips.perceptual import PerceptualLoss


def custom_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        print("Warning - matrics = custom_psnr - PSNR = 100 !!! ")
        return 100
    max_pixel = 1.
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def determine_original_sliceids(reference, downsample_steps, conv_interpol=False):
    orig_num_slices = reference.shape[0]
    slice_ids = np.arange(orig_num_slices)
    keep_slice_ids = None
    if (orig_num_slices - 1) % downsample_steps != 0:
        remain_num_slices = (orig_num_slices - 1) % downsample_steps
        keep_slice_ids = slice_ids[-remain_num_slices:]
        slice_ids = slice_ids[:-remain_num_slices]
    if conv_interpol and slice_ids.shape[0] % downsample_steps != 0:
        remain_num_slices = slice_ids.shape[0] % downsample_steps
        keep_slice_ids = slice_ids[-remain_num_slices:] if keep_slice_ids is None else \
            np.concatenate((slice_ids[-remain_num_slices:], keep_slice_ids))
        slice_ids = slice_ids[:-remain_num_slices]
    slice_ids = slice_ids[::downsample_steps]
    if keep_slice_ids is not None:
        slice_ids = np.concatenate((slice_ids, keep_slice_ids))
    return slice_ids


def check_image_object_type(l_images, l_reconstructions):
    if not isinstance(l_images, np.ndarray):
        # cast from pytorch
        l_images = l_images.detach().cpu().numpy()
    if not isinstance(l_reconstructions, np.ndarray):
        l_reconstructions = l_reconstructions.detach().cpu().numpy()
    l_images = l_images.astype(np.float32)
    l_reconstructions = l_reconstructions.astype(np.float32)

    # assuming images and reconstruction have shape [b, c, w, h]
    if l_images.ndim > 3:
        l_images = np.squeeze(l_images)
    if l_reconstructions.ndim > 3:
        l_reconstructions = np.squeeze(l_reconstructions)
    return l_images, l_reconstructions


def compute_vif_for_batch(l_images, l_reconstructions, eval_axis=0, normalize=False, downsample_steps=None,
                           conv_interpol=False):
    l_images, l_reconstructions = check_image_object_type(l_images, l_reconstructions)
    # we assume l_images is always normalized, but our reconstruction not always
    if normalize:
        l_reconstructions = rescale_intensities(l_reconstructions, percs=tuple((0, 100)))

    l_images = np.uint8(np.clip(l_images * 255., 0, 255))
    l_reconstructions = np.uint8(np.clip(l_reconstructions * 255., 0, 255))
    # we assume that after squeeze we have 2 dim shape [w, h] or 3 dim shape [z, w, h]
    if l_images.ndim == 3:
        if downsample_steps is not None:
            print("WARNING - compute_vif_for_batch - Downsample")
            orig_slice_ids = determine_original_sliceids(l_images, downsample_steps=downsample_steps,
                                                         conv_interpol=conv_interpol)
        else:
            orig_slice_ids = []
        invalid_vif = 0
        res = []
        if eval_axis != 0:
            l_images = copy.deepcopy(l_images)
            l_reconstructions = copy.deepcopy(l_reconstructions)
            l_images = np.swapaxes(l_images, 0, eval_axis)
            l_reconstructions = np.swapaxes(l_reconstructions, 0, eval_axis)
        for z in np.arange(l_images.shape[0]):
            # vif_score = vifp(l_images[z], l_reconstructions[z])
            if downsample_steps is not None:
                if z in orig_slice_ids:
                    # dealing with original slice, skip for evaluation
                    continue
            # when evaluating cross-sectional images, we do not want to evaluate on black images
            if eval_axis != 0:
                if np.sum(l_images[z]) == 0:
                    continue
            vif_score = vifp_mscale(l_images[z], l_reconstructions[z])  # vifp  vifp_mscale
            # vif_score = vifvec_alternative(l_images[z], l_reconstructions[z])
            if not np.isnan(vif_score) and not np.isinf(vif_score):
                res.append(vif_score)
            else:
                invalid_vif += 1

        return np.mean(np.array(res))
    else:
        return vifp_mscale(l_images, l_reconstructions)


def compute_ssim_for_batch(l_images, l_reconstructions, eval_axis=0, normalize=False, downsample_steps=None,
                           conv_interpol=False):

    l_images, l_reconstructions = check_image_object_type(l_images, l_reconstructions)
    # we assume l_images is always normalized, but our reconstruction not always
    if normalize:
        l_reconstructions = rescale_intensities(l_reconstructions, percs=tuple((0, 100)))
    # we assume that after squeeze we have 2 dim shape [w, h] or 3 dim shape [z, w, h]
    if l_images.ndim == 3:
        if downsample_steps is not None:
            orig_slice_ids = determine_original_sliceids(l_images, downsample_steps=downsample_steps,
                                                         conv_interpol=conv_interpol)
        else:
            orig_slice_ids = []
        res = []
        if eval_axis != 0:
            # print("SSIM, before ", l_images.shape, l_reconstructions.shape)
            l_images = copy.deepcopy(l_images)
            l_reconstructions = copy.deepcopy(l_reconstructions)
            l_images = np.swapaxes(l_images, 0, eval_axis)
            l_reconstructions = np.swapaxes(l_reconstructions, 0, eval_axis)
            # print("SSIM, after ", l_images.shape, l_reconstructions.shape)
        for z in np.arange(l_images.shape[0]):
            if downsample_steps is not None:
                if z in orig_slice_ids:
                    # dealing with original slice, skip for evaluation
                    continue
            if eval_axis == 0:
                res.append(ssim_metric(l_images[z], l_reconstructions[z]))
            else:
                # ONLY NECESSARY IF WE EVALUATE LONG AXIS VIEWS
                # when evaluating cross-sectional images, we do not want to evaluate on black images
                if eval_axis != 0:
                    if np.sum(l_images[z]) == 0:
                        continue
                x_size, y_size = l_images[z].shape
                if x_size < 8 or y_size < 8:
                    # This is hacky but necessary for evaluation of long axis views if we have less than 8 slices.
                    # then l_images[z] will be a slice with less than size 8 in dim0 or 1. win_size needs to be an odd
                    # number.
                    res.append(ssim_metric(l_images[z], l_reconstructions[z], win_size=5))
                else:
                    res.append(ssim_metric(l_images[z], l_reconstructions[z]))
        return np.mean(np.array(res))
    else:
        return ssim_metric(l_images, l_reconstructions)


def compute_psnr_for_batch(l_images, l_reconstructions, eval_axis=0, normalize=False, downsample_steps=None,
                           conv_interpol=False):
    # assuming images and reconstruction have shape [b, c, w, h]
    l_images, l_reconstructions = check_image_object_type(l_images, l_reconstructions)
    # we assume l_images is always normalized, but our reconstruction not always
    if normalize:
        l_reconstructions = rescale_intensities(l_reconstructions, percs=tuple((0, 100)))
    # we assume that after squeeze we have 2 dim shape [w, h] or 3 dim shape [z, w, h]
    if l_images.ndim == 3:
        if downsample_steps is not None:
            orig_slice_ids = determine_original_sliceids(l_images, downsample_steps=downsample_steps,
                                                         conv_interpol=conv_interpol)
        else:
            orig_slice_ids = []
        res = []
        if eval_axis != 0:
            l_images = copy.deepcopy(l_images)
            l_reconstructions = copy.deepcopy(l_reconstructions)
            l_images = np.swapaxes(l_images, 0, eval_axis)
            l_reconstructions = np.swapaxes(l_reconstructions, 0, eval_axis)
        for z in np.arange(l_images.shape[0]):
            if downsample_steps is not None:
                if z in orig_slice_ids:
                    # dealing with original slice, skip for evaluation
                    continue
            # when evaluating cross-sectional images, we do not want to evaluate on black images
            if eval_axis != 0:
                if np.sum(l_images[z]) == 0:
                    continue
            psnr = psnr_metric(l_images[z], l_reconstructions[z])
            # psnr = custom_psnr(l_images[z], l_reconstructions[z])
            if not np.isnan(psnr) and not np.isinf(psnr):
                res.append(psnr)
        return np.mean(np.array(res))
    else:
        return psnr_metric(l_images, l_reconstructions)


def check_image_is_cuda(l_images, l_reconstructions):
    if isinstance(l_images, np.ndarray):
        l_images = torch.FloatTensor(l_images)
    if isinstance(l_reconstructions, np.ndarray):
        l_reconstructions = torch.FloatTensor(l_reconstructions)
    if not l_images.cuda:
        l_images = l_images.to('cuda:1')
    if not l_reconstructions.cuda:
        l_reconstructions = l_reconstructions.to('cuda:1')

    return l_images, l_reconstructions


def compute_lpips_for_batch(l_images, l_reconstructions, eval_axis=0, normalize=False, downsample_steps=None,
                            conv_interpol=False, criterion=None):
    if criterion is None:
        criterion = PerceptualLoss(
            model='net-lin', net='vgg', use_gpu='cuda', gpu_ids=[0])
    # assuming images and reconstruction have shape [b, c, w, h]
    l_images, l_reconstructions = check_image_object_type(l_images, l_reconstructions)
    # we assume l_images is always normalized, but our reconstruction not always
    if normalize:
        l_reconstructions = rescale_intensities(l_reconstructions, percs=tuple((0, 100)))
    # we assume that after squeeze we have 2 dim shape [w, h] or 3 dim shape [z, w, h]
    if l_images.ndim == 3:
        if downsample_steps is not None:
            orig_slice_ids = determine_original_sliceids(l_images, downsample_steps=downsample_steps,
                                                         conv_interpol=conv_interpol)
        else:
            orig_slice_ids = []
        res = []
        if eval_axis != 0:
            l_images = copy.deepcopy(l_images)
            l_reconstructions = copy.deepcopy(l_reconstructions)
            l_images = np.swapaxes(l_images, 0, eval_axis)
            l_reconstructions = np.swapaxes(l_reconstructions, 0, eval_axis)
        for z in np.arange(l_images.shape[0]):
            if downsample_steps is not None:
                if z in orig_slice_ids:
                    # dealing with original slice, skip for evaluation
                    continue
            image_slice, recon_slice = check_image_is_cuda(l_images[z], l_reconstructions[z])
            lpips_metric = criterion(image_slice, recon_slice, normalize=True)
            res.append(lpips_metric.item())
        return np.mean(np.array(res))
    else:
        return criterion(l_images, l_reconstructions)