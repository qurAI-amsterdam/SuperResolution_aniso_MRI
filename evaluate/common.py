import torch
import SimpleITK as sitk
import copy
import os
from torchvision.utils import make_grid
from scipy.ndimage import gaussian_filter
from kwatsch.img_interpolation import latent_space_interp
import numpy as np


def save_metrics(output_dir: str, eval_dataset: str, metrics_dict: dict,
                 downsample_steps: int, interpol_method: str, eval_axis: int) -> None:

    output_dir = os.path.join(output_dir, "results")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=False)
    if eval_axis == 0:
        file_name = "{}_{}x.npz".format(interpol_method, downsample_steps)
    else:
        file_name = "{}_{}x_axis{}.npz".format(interpol_method, downsample_steps, eval_axis)
    if eval_dataset is not None:
        file_name = "{}_".format(eval_dataset) + file_name
    abs_file_name = os.path.join(output_dir, file_name)

    np.savez(abs_file_name, **metrics_dict)
    print("INFO - Saved results to {}".format(abs_file_name))


def strip_conventional_interpolation_results(img3d_sr, img3d_original, expand_factor):
    # expand factor is equal to num_interpolations + 1
    # we strip the last expand_factor slices and replace with original slice
    # because that is then equivalent with SR interpolation results
    return np.concatenate((img3d_sr[:-expand_factor], img3d_original[-1][None]))


def determine_last_slice(orig_num_slices, downsample_steps):

    last_slice = np.arange(orig_num_slices)[::downsample_steps][-1]
    return last_slice


def eval_on_different_patch_size(trainer, test_images, patch_size=tuple((32, 32))):
    # assuming torch tensor on cpu with shape [z, y, x]
    assert isinstance(patch_size, tuple) or isinstance(patch_size, int)
    if not isinstance(patch_size, tuple):
        patch_size = tuple((patch_size, patch_size))
    recon_vol = None
    for z in np.arange(test_images.shape[0]):
        recon = create_recon_from_diff_psize(trainer, test_images[z], patch_size)
        recon_vol = torch.cat([recon_vol, recon[None]]) if recon_vol is not None else recon[None]
    return recon_vol


def create_recon_from_diff_psize(trainer, test_images, patch_size=tuple((32, 32))):
    # Assuming that test_images is one image of size
    if test_images.dim() > 2:
        test_images = torch.squeeze(test_images)
    w, h = test_images.shape
    new_batch_size = (w // patch_size[0]) * (h // patch_size[1])
    img_patches = test_images.unfold(0, *patch_size).unfold(1, *patch_size)
    img_patches = torch.reshape(img_patches, (new_batch_size, 1, patch_size[0], patch_size[1]))
    recons = trainer.predict(img_patches.to('cuda'))
    recons = make_grid(recons, int(np.sqrt(recons.size(0))), 0)[0]
    return recons.detach().cpu()


def generate_blurred_sr_image(image: sitk.Image, abs_filename: str):
    blurred_simple_img = apply_blur_filter(sitk.GetArrayFromImage(image).astype(np.float32), sigma=1.25)
    blurred_simple_img = sitk.GetImageFromArray(blurred_simple_img)
    blurred_simple_img.SetSpacing(image.GetSpacing())
    sitk.WriteImage(blurred_simple_img, abs_filename.replace(".nii.gz", "_blurred_lanczoso.nii.gz"), False)


def create_simple_interpolation(images: np.ndarray, spacing, new_spacing_z=None, expand_factor=None,
                                interpol_filter=sitk.sitkLanczosWindowedSinc,
                                generate_inbetween_slices=False):
    assert not (new_spacing_z is None and expand_factor is None)
    if not isinstance(images, np.ndarray):
        images = images.detach().cpu().numpy()

    l_spacing = copy.deepcopy(spacing)
    if expand_factor is None:
        expand_factor = int(np.ceil(l_spacing[0] / new_spacing_z))
    else:
        expand_factor = int(expand_factor)
    if generate_inbetween_slices:
        downsample_steps = expand_factor
        orig_image, orig_num_slices = copy.deepcopy(images), images.shape[0]
        last_slice_id = determine_last_slice(orig_num_slices, downsample_steps)
        images = images[::downsample_steps]
        l_spacing[0] = expand_factor * l_spacing[0]
    # Upsample images
    sitk_img = sitk.GetImageFromArray(images)
    sitk_img.SetSpacing(l_spacing[::-1].astype(np.float64))
    new_images_simple = simple_interploation(sitk_img, expand_factor=expand_factor,
                                             interpol_filter=interpol_filter)
    if generate_inbetween_slices:
        np_new_img = sitk.GetArrayFromImage(new_images_simple)
        if (orig_num_slices - 1) % downsample_steps == 0:
            np_new_img = np_new_img[:(last_slice_id + 1)]
        else:
            remain_num_slices = (orig_num_slices - 1) % downsample_steps
            np_new_img = np_new_img[:last_slice_id + 1]
            if remain_num_slices > 0:
                keep_slices = orig_image[-remain_num_slices:]
                np_new_img = np.concatenate((np_new_img, keep_slices), axis=0)
        t_img = sitk.GetImageFromArray(np_new_img)
        t_img.SetSpacing(new_images_simple.GetSpacing())
        t_img.SetOrigin((new_images_simple.GetOrigin()))
        new_images_simple = t_img
    return new_images_simple


def simple_interploation(img: sitk.Image, expand_factor, interpol_filter=sitk.sitkLanczosWindowedSinc) -> sitk.Image:
    img_expander = sitk.ExpandImageFilter()
    img_expander.SetInterpolator(interpol_filter)
    img_expander.SetExpandFactors((1, 1, expand_factor))
    return img_expander.Execute(img)


def apply_blur_filter(img: np.ndarray, sigma=1.) -> np.ndarray:
    new_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        new_img[i] = gaussian_filter(img[i], sigma)
    return new_img


def rescale_tensor(p_tensor):
    p_tensor = (
                (p_tensor - torch.min(p_tensor)) / (torch.max(p_tensor) - torch.min(p_tensor)))
    return torch.clamp(p_tensor, min=0, max=1)


def create_super_volume(trainer, images, alpha_range=None, use_original=False, hierarchical=False,
                        downsample_steps=None, generate_inbetween_slices=False,
                        train_patch_size=None, feature_dict=None, labels=None):
    """

    :param trainer: Assuming images is tensor of patient volume with shape [z, 1, y, x] or [z, y, x]

    :param images:
    :param alpha_range:
    :param use_original: boolean, create sr volume with original slices that are not reconstructed or reconstruct all slices
    :param generate_inbetween_slices
    :param train_patch_size: training patch size is smaller than patch size we evaluate on
    :return:
    """
    torch.cuda.empty_cache()
    with_labels = False if labels is None else True
    if generate_inbetween_slices and downsample_steps is None:
        downsample_steps = int(len(alpha_range) + 1)

    num_slices, w, h = images.size()
    orig_images, orig_num_slices = None, images.size(0)

    if not generate_inbetween_slices and downsample_steps is not None:
        # IMPORTANT DOWNSAMPLE ONLY FOR GENERATE_INBETWEEN SLICES!!!!! But may be sometimes for foefelen you want this
        print("WARNING !!! create_super_volume - downsample steps is not None but generate_inbetween_slices is False!")
    if downsample_steps is not None or generate_inbetween_slices:
        orig_images = copy.deepcopy(images)
        orig_labels = None if not with_labels else copy.deepcopy(labels)
        if (orig_num_slices - 1) % downsample_steps != 0:
            remain_num_slices = (orig_num_slices - 1) % downsample_steps
            images = images[:-remain_num_slices]
            labels = None if labels is None else labels[:-remain_num_slices]

        images = images[::downsample_steps]
        labels = None if labels is None else labels[::downsample_steps]
        num_slices = images.size(0)

    if images.dim() == 3:
        # make sure [z, 1, y, x]
        images = torch.unsqueeze(images, dim=1)
    if with_labels and labels.dim() == 3:
        labels = torch.unsqueeze(labels, dim=1)
    images = images if labels is None else torch.cat([images, labels], dim=1)
    if alpha_range is None:
        alpha_range = [0.25, 0.5, 0.75]

    if not use_original:
        if images.dtype != torch.cuda.FloatTensor:
            images = images.to('cuda')
        recon_dict = trainer.predict(images)
        if with_labels:
            recon_volume = recon_dict['image'].detach().cpu()
            recon_labels = recon_dict['pred_labels'].detach().cpu()

        else:
            recon_volume = recon_dict.detach().cpu()
            recon_labels = None
    else:
        recon_volume = images
        recon_labels = None if not with_labels else labels
    images2 = images[1:]
    images1 = images[:-1]
    interp_slices, interp_slice_labels = None, None
    img_pred_alphas = None
    # print("WARNING - create_super_volume - ", generate_inbetween_slices, downsample_steps, images.shape)
    for i, alpha in enumerate(alpha_range):
        inter_result_dict = latent_space_interp(alpha, trainer, images2, images1, hierarchical=hierarchical,
                                                with_labels=with_labels)
        interp_img = inter_result_dict['inter_image']
        inter_label = None if not with_labels else inter_result_dict['inter_label']
        pred_alphas = torch.FloatTensor([alpha]).expand_as(images2)
        img_pred_alphas = torch.cat((img_pred_alphas, pred_alphas), dim=0) if img_pred_alphas is not None else pred_alphas
        interp_slices = interp_img if interp_slices is None else torch.cat([interp_slices, interp_img], dim=1)
        if with_labels:
            interp_slice_labels = inter_label if interp_slice_labels is None else torch.cat([interp_slice_labels, inter_label], dim=1)
    # interp_slices = rescale_tensor(interp_slices)
    new_volume, new_labels = None, None
    for i in range(num_slices - 1):
        new_volume = torch.cat([recon_volume[i], interp_slices[i]]) if new_volume is None else torch.cat \
                ([new_volume, recon_volume[i], interp_slices[i]], dim=0)
        if with_labels:
            new_labels = torch.cat([recon_labels[i], interp_slice_labels[i]]) if new_labels is None else torch.cat \
                ([new_labels, recon_labels[i], interp_slice_labels[i]], dim=0)
    # add last slice
    new_volume = torch.cat([new_volume, recon_volume[i + 1]])
    new_labels = None if not with_labels else torch.cat([new_labels, recon_labels[i + 1]])
    if generate_inbetween_slices and (orig_num_slices - 1) % downsample_steps != 0:
        remain_num_slices = (orig_num_slices - 1) % downsample_steps
        remaining_slices = orig_images[-remain_num_slices:]
        remain_slices_labels = None if not with_labels else orig_labels[-remain_num_slices:]
        if remaining_slices.dim() == 2:
            remaining_slices = remaining_slices[:, None]
            remain_slices_labels = None if not with_labels else remain_slices_labels[:, None]
        # we create every 2nd slice by means of interpolation. If the original slice number is even
        # we're not interpolating between the penultimate and last slice. Hence, here we add the last
        # slice of the original stack, note, this is the reconstruction of the original at least
        new_volume = torch.cat([new_volume, remaining_slices])
        new_labels = None if not with_labels else torch.cat([new_labels, remain_slices_labels])

    new_volume = torch.clamp(new_volume, min=0, max=1.)

    return {'upsampled_image': new_volume, 'upsampled_labels': new_labels, 'pred_alphas': img_pred_alphas}





