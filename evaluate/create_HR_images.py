import os
import numpy as np
import torch
import types
import copy
from pathlib import Path
import SimpleITK as sitk
from kwatsch.alpha.common import get_alpha_probe_features
from evaluate.common import create_super_volume, create_simple_interpolation
from evaluate.metrics import compute_ssim_for_batch, compute_psnr_for_batch, compute_vif_for_batch
from evaluate.metrics import compute_lpips_for_batch
from datasets.ACDC.data import sitk_save
import cv2
from lpips.perceptual import PerceptualLoss
from evaluate.common import determine_last_slice
from evaluate.quantitative_comparison import generate_synth_slices_mask
from datasets.ARVC.sr_data_with_labels import translate_labels
from datasets.common import apply_2d_zoom_4d, apply_2d_zoom_3d


FILTER_BENCH_SITK = {'lanczos': sitk.sitkLanczosWindowedSinc,
                'nearest': sitk.sitkNearestNeighbor,
                'bspline': sitk.sitkBSpline,
                'linear': sitk.sitkLinear}

FILTER_BENCH_CV2 = {'lanczos': cv2. INTER_LANCZOS4,
                'nearest': cv2.INTER_NEAREST,
                    'cubic': cv2.INTER_CUBIC,
                'linear': cv2.INTER_LINEAR}


def get_percept_loss_criterion(trainer):
    percept_loss = None
    if trainer is not None:
        if trainer.percept_criterion is not None:
            percept_loss = trainer.percept_criterion
    if percept_loss is None:
        percept_loss = PerceptualLoss(
            model='net-lin', net='vgg', use_gpu='cuda', gpu_ids=[1])
    return percept_loss


def init_create_hr_images(trainer, myargs, num_interpolations, downsample_steps, interpol_filter,
                          generate_inbetween_slices, compute_percept_loss, base_out_dir):

    assert (num_interpolations is not None or downsample_steps is not None)
    if hasattr(trainer, "alpha_probe") and trainer.alpha_probe is not None:
        feature_dict = {'anatomy': 'cardiac' if myargs['dataset'] in ['ACDC', 'ARVC', 'ACDCC'] else 'brain'}
    else:
        feature_dict = None
    if num_interpolations is not None and downsample_steps is not None and interpol_filter is None:
        # we're using our proposed model to generate HR volumes
        if generate_inbetween_slices and num_interpolations + 1 != downsample_steps:
            raise ValueError("ERROR - num_interpolations {} must be equal to "
                             "downsample_steps {} - 1".format(num_interpolations, downsample_steps))
    if compute_percept_loss:
        percept_loss = get_percept_loss_criterion(trainer)
    else:
        percept_loss = None
    if base_out_dir is None and generate_inbetween_slices:
        base_out_dir = "images_sr" if interpol_filter is None else interpol_filter
    else:
        base_out_dir = "images_sr_ip" if interpol_filter is None else interpol_filter
    if interpol_filter is not None:
        interpol_filter = FILTER_BENCH_SITK[interpol_filter]
    if generate_inbetween_slices and interpol_filter is not None and downsample_steps is None:
        raise ValueError("ERROR - parameter downsample_steps must be specified in when using conventional"
                         " interpolation method for generating in-between-slices.")
    return interpol_filter, feature_dict, base_out_dir, percept_loss


def check_data_generator(data_generator):
    if not isinstance(data_generator, types.GeneratorType):
        if isinstance(data_generator, dict):
            data_generator = data_generator.values()
        else:
            raise ValueError("ERROR - create_hr_images - data_generator is not a generator nor a dict")
    return data_generator


def save_3d_volume(new_images, pat_id, output_dir, file_suffix, sr_spacing, origin, direction, myargs,
                   original_spacing,  resample=False, is_arvc_labels=False, new_labels=None, output_dir_labels=None):
    if resample:
        new_images = apply_2d_zoom_3d(new_images, sr_spacing, original_spacing, do_blur=True,
                                         as_type=np.float32)
        # Important we need to set new sr spacing: Assuming shape [z, y, x] spacing. We keep z-spacing.
        sr_spacing[1:] = original_spacing[1:]
    pat_output_dir = os.path.join(output_dir, pat_id)
    if not os.path.isdir(pat_output_dir):
        os.makedirs(pat_output_dir, exist_ok=False)
    if file_suffix is None:
        filename = pat_id + "_{}".format(myargs['model']) + ".nii.gz"
    else:
        filename = pat_id + "_{}".format(file_suffix) + ".nii.gz"
    abs_filename = os.path.join(pat_output_dir, filename)
    sitk_save(abs_filename, new_images, spacing=sr_spacing, dtype=np.float32, normalize=False,
              origin=origin, direction=direction)
    if new_labels is not None:
        if is_arvc_labels:
            new_labels = translate_labels(new_labels, "acdc_to_arvc")
        pat_output_dir = os.path.join(output_dir_labels, pat_id)
        if not os.path.isdir(pat_output_dir):
            os.makedirs(pat_output_dir, exist_ok=False)
        abs_filename = os.path.join(pat_output_dir, filename)
        sitk_save(abs_filename, new_labels, spacing=sr_spacing, dtype=np.int32, normalize=False,
                  origin=origin, direction=direction)
    print("INFO - saved {}".format(abs_filename))


def compute_mean_metrics(ssim_results, psnr_results, vif_results, lpips_results,
                         compute_percept_loss=False):
    mean_ssim, std_ssim = np.mean(np.array(ssim_results)), np.std(np.array(ssim_results))
    mean_psnr, std_psnr = np.mean(np.array(psnr_results)), np.std(np.array(psnr_results))
    mean_vif, std_vif = np.mean(np.array(vif_results)), np.std(np.array(vif_results))
    mean_lpips, std_lpips = 0, 0
    if compute_percept_loss:
        mean_lpips, std_lpips = np.mean(np.array(lpips_results)), np.std(np.array(lpips_results))
    return mean_ssim, std_ssim, mean_psnr, std_psnr, mean_vif, std_vif, mean_lpips, std_lpips


def compute_metrics(images_ref, new_images, downsample_steps,
                    ssim_results, psnr_results, vif_results, lpips_results,
                    ssim_res_synth=None, psnr_res_synth=None, vif_res_synth=None, lpips_res_synth=None,
                    ssim_res_recon=None, psnr_res_recon=None, vif_res_recon=None, lpips_res_recon=None,
                    compute_percept_loss=False,
                    percept_loss=None,
                    normalize=False, eval_axis=0):

    def compute_masked_metrics(images_ref, new_images, slice_mask, last_slice_id, ssim_res, psnr_res, vif_res, lpips_res,
                               compute_percept_loss=False, percept_loss=None):
        ssim_res.append(
            compute_ssim_for_batch(images_ref[:last_slice_id][slice_mask],
                                   new_images[:last_slice_id][slice_mask], normalize=normalize, eval_axis=eval_axis))
        psnr_res.append(
            compute_psnr_for_batch(images_ref[:last_slice_id][slice_mask],
                                   new_images[:last_slice_id][slice_mask], normalize=normalize, eval_axis=eval_axis))
        vif_res.append(
            compute_vif_for_batch(images_ref[:last_slice_id][slice_mask],
                                  new_images[:last_slice_id][slice_mask], normalize=normalize, eval_axis=eval_axis))
        if compute_percept_loss:
            lpips_res.append(compute_lpips_for_batch(
                images_ref[:last_slice_id][slice_mask],
                new_images[:last_slice_id][slice_mask], normalize=normalize,
                criterion=percept_loss))
        return ssim_res, psnr_res, vif_res, lpips_res

    # we need to add 1 to last slice id because we want to compute metrics tot/met this slice
    last_slice_id = determine_last_slice(images_ref.shape[0], downsample_steps) + 1
    r_mask, s_mask = generate_synth_slices_mask(images_ref.shape[0], downsample_steps=downsample_steps)
    ssim_results.append(
        compute_ssim_for_batch(images_ref[:last_slice_id],
                               new_images[:last_slice_id], normalize=normalize, eval_axis=eval_axis))
    psnr_results.append(
        compute_psnr_for_batch(images_ref[:last_slice_id],
                               new_images[:last_slice_id], normalize=normalize, eval_axis=eval_axis))
    vif_results.append(
        compute_vif_for_batch(images_ref[:last_slice_id],
                              new_images[:last_slice_id], normalize=normalize, eval_axis=eval_axis))
    if compute_percept_loss:
        lpips_results.append(compute_lpips_for_batch(
            images_ref[:last_slice_id],
            new_images[:last_slice_id], normalize=normalize,
            criterion=percept_loss))
    if ssim_res_synth is not None and eval_axis == 0:
        ssim_res_synth, psnr_res_synth, vif_res_synth, lpips_res_synth = \
            compute_masked_metrics(images_ref, new_images, s_mask, last_slice_id, ssim_res_synth,
                                   psnr_res_synth, vif_res_synth,
                                   lpips_res_synth, compute_percept_loss=False, percept_loss=percept_loss)

    if ssim_res_recon is not None and eval_axis == 0:
        ssim_res_recon, psnr_res_recon, vif_res_recon, lpips_res_recon = \
            compute_masked_metrics(images_ref, new_images, r_mask, last_slice_id, ssim_res_recon,
                                   psnr_res_recon, vif_res_recon,
                                   lpips_res_recon, compute_percept_loss=False, percept_loss=percept_loss)

    return ssim_results, psnr_results, vif_results, lpips_results,  \
            ssim_res_synth, psnr_res_synth, vif_res_synth, lpips_res_synth, \
                ssim_res_recon, psnr_res_recon, vif_res_recon, lpips_res_recon


def save_metrics_to_file(result_dict, fname):
    metrics = {'ssim': np.array(result_dict['ssim']), 'psnr': np.array(result_dict['psnr']),
                    'vif': np.array(result_dict['vif']), 'lpips': np.array(result_dict['lpips']),
                    # same for synthesized only results
                    'ssim_synth': np.array(result_dict['ssim_synth']),
                    'psnr_synth': np.array(result_dict['psnr_synth']),
                    'vif_synth': np.array(result_dict['vif_synth']),
                    'lpips_synth': np.array(result_dict['lpips_synth']),
                    # reconstructed
                    'ssim_recon': np.array(result_dict['ssim_recon']),
                    'psnr_recon': np.array(result_dict['psnr_recon']),
                    'vif_recon': np.array(result_dict['vif_recon']),
                    'lpips_recon': np.array(result_dict['lpips_recon']),
                    }
    np.savez(fname, **metrics)
    print("INFO - Saved results to {}".format(fname))


def save_4d_volume(new_4d_volume, save_patid, output_dir, file_suffix, save_sr_spacing, save_origin, save_direction,
                   myargs, save_original_spacing, resample=False, is_arvc_labels=False, new_4d_labels=None,
                   output_dir_labels=None):
    spacing_img = copy.deepcopy(save_sr_spacing)
    spacing_lbl = copy.deepcopy(save_sr_spacing)
    if resample:
        new_4d_volume = apply_2d_zoom_4d(new_4d_volume, save_sr_spacing, save_original_spacing, do_blur=True,
                                         as_type=np.float32)
        # Important we need to set new sr spacing: Assuming shape [z, y, x] spacing. We keep z-spacing.
        spacing_img[1:] = save_original_spacing[1:]
        print("Create hr images - resampled IMAGE - original/new spacing ", save_original_spacing, spacing_img)
    else:
        print("Create hr images - using spacing ", spacing_img, new_4d_volume.shape)
    pat_output_dir = os.path.join(output_dir, save_patid)
    if not os.path.isdir(pat_output_dir):
        os.makedirs(pat_output_dir, exist_ok=False)
    if file_suffix is None:
        filename = save_patid + "_4d_{}".format(myargs['model']) + ".nii.gz"
    else:
        filename = save_patid + "_{}".format(file_suffix) + ".nii.gz"
    abs_filename = os.path.join(pat_output_dir, filename)
    sitk_save(abs_filename, new_4d_volume, spacing=spacing_img, dtype=np.float32,
              origin=save_origin, direction=save_direction, normalize=False)
    if new_4d_labels is not None:
        if is_arvc_labels:
            new_4d_labels = translate_labels(new_4d_labels, "acdc_to_arvc")
        if resample:
            new_4d_labels = apply_2d_zoom_4d(new_4d_labels, save_sr_spacing, save_original_spacing, do_blur=False,
                                             order=0, as_type=np.int)
            spacing_lbl[1:] = save_original_spacing[1:]
            print("Create hr images - resampled LABELS - original/new spacing ", save_original_spacing, spacing_lbl)
        pat_output_dir = os.path.join(output_dir_labels, save_patid)
        if not os.path.isdir(pat_output_dir):
            os.makedirs(pat_output_dir, exist_ok=False)
        abs_filename = os.path.join(pat_output_dir, filename)
        sitk_save(abs_filename, new_4d_labels, spacing=spacing_lbl, dtype=np.int32, normalize=False,
                  origin=save_origin, direction=save_direction)
    print("INFO - saved {}".format(abs_filename))


def create_hr_images(data_generator, myargs, trainer=None, num_interpolations=None, downsample_steps=None,
                     is_4d=False, transform=None, expand_factor=None, use_original_slice=False,
                     normalize=False, generate_inbetween_slices=False, file_suffix=None, patient_id=None,
                     interpol_filter=None, save_volumes=False, output_dir=None, verbose=False,
                     compute_percept_loss=False, base_out_dir=None, eval_axis=0, is_arvc_labels=False,
                     resample=False):
    """
    """
    interpol_filter, feature_dict, base_out_dir, percept_loss = init_create_hr_images(trainer, myargs,
                                                                                      num_interpolations,
                                                                                      downsample_steps, interpol_filter,
                                                                                      generate_inbetween_slices,
                                                                                      compute_percept_loss,
                                                                                      base_out_dir)
    if num_interpolations is not None:
        alpha_range = np.linspace(0, 1, num_interpolations + 2, endpoint=True)[1:-1]
    else:
        alpha_range = None
    if output_dir is not None:
        output_dir = os.path.join(os.path.expanduser(output_dir), base_out_dir)
    else:
        assert 'output_dir' in myargs
        output_dir = os.path.join(myargs['output_dir'], base_out_dir)
    output_dir_labels = output_dir.replace("images_sr", "labels_sr")
    if save_volumes:
        print("INFO - saving output to {}".format(output_dir))
    if save_volumes and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=False)

    data_generator = check_data_generator(data_generator)

    save_patid = None
    save_sr_spacing, save_origin, save_direction = None, None, None
    ssim_results, psnr_results, vif_results, lpips_results = [],  [], [], []
    ssim_res_synth, psnr_res_synth, vif_res_synth, lpips_res_synth = [],  [], [], []
    ssim_res_recon, psnr_res_recon, vif_res_recon, lpips_res_recon = [], [], [], []
    for test_batch in data_generator:
        origin = test_batch['origin'] if 'origin' in test_batch.keys() else None
        direction = test_batch['direction'] if 'direction' in test_batch.keys() else None
        original_spacing = test_batch['original_spacing'] if 'original_spacing' in test_batch.keys() else None
        if transform is not None:
            test_batch = transform(test_batch)
        images = torch.from_numpy(test_batch['image'])
        labels = None if 'labels' not in test_batch.keys() else torch.from_numpy(transform({'image': test_batch['labels']})['image'])
        if 'image_hr' in test_batch.keys() and test_batch['image_hr'] is not None:
            image_hr = test_batch['image_hr'] if transform is None else transform({'image': test_batch['image_hr']})['image']
        else:
            image_hr = None
        # images is [z, y, x], if train patch size is different then x or y (assume squared patch size) then
        # we are evaluating on a bigger patch than we trained on
        train_patch_size = None
        # 07-12-2020 Disabled because it looks like we can train with smaller patch size and evaluate large
        # without problems if compression factor stays the same
        # if interpol_filter is None and myargs['width'] != images.shape[2]:
        #    train_patch_size = tuple((myargs['width'], myargs['width']))
        pat_id = test_batch['patient_id'] if isinstance(test_batch['patient_id'], str) else str(test_batch['patient_id'])
        if patient_id is not None:
            # if patient_id (parameter) is not None then skip all other patients from generator
            if patient_id != pat_id:
                continue

        if generate_inbetween_slices:
            # IMPORTANT: we assume in this case that the original high resolution image is reconstructed, so
            # resolution will not change.
            new_spacing_z = test_batch['spacing'][0]
        else:
            new_spacing_z = test_batch['spacing'][0] / (num_interpolations + 1)
        if interpol_filter is None:
            if feature_dict is not None:
                feature_dict = get_alpha_probe_features(feature_dict, test_batch)
            create_super_volume_dict = create_super_volume(trainer, images, alpha_range=alpha_range, use_original=use_original_slice,
                                             downsample_steps=downsample_steps, hierarchical=False,
                                             generate_inbetween_slices=generate_inbetween_slices,
                                             train_patch_size=train_patch_size, feature_dict=feature_dict, labels=labels)
            new_images, pred_alphas = create_super_volume_dict['upsampled_image'], \
                                        create_super_volume_dict['pred_alphas']
            new_labels = None if labels is None else create_super_volume_dict['upsampled_labels']
            # copy spacing
            sr_spacing = test_batch['spacing'][:].astype(np.float64)
        else:
            # conventional interpolation methods
            if not generate_inbetween_slices:
                assert expand_factor is not None, "For conventional method expand factor must be specified."
            new_images = create_simple_interpolation(images, test_batch['spacing'], new_spacing_z,
                                                     expand_factor=expand_factor,
                                                     interpol_filter=interpol_filter,
                                                     generate_inbetween_slices=generate_inbetween_slices)
            sr_spacing = np.array(new_images.GetSpacing()[::-1]).astype(np.float64)
            new_images = sitk.GetArrayFromImage(new_images)
            new_images = new_images.clip(0, 1)
            new_labels = None
        # set new z-spacing
        sr_spacing[0] = new_spacing_z
        # sitk_save expects numpy array
        if not isinstance(new_images, np.ndarray):
            new_images = new_images.detach().cpu().numpy()
        if new_labels is not None and isinstance(new_images, np.ndarray):
            new_labels = new_labels.detach().cpu().numpy()
        if not is_4d and save_volumes:
            save_3d_volume(new_images, pat_id, output_dir, file_suffix, sr_spacing, origin, direction, myargs,
                           original_spacing, resample=resample, is_arvc_labels=is_arvc_labels, new_labels=new_labels,
                           output_dir_labels=output_dir_labels)

        if is_4d and save_volumes and (save_patid is None or pat_id != save_patid):
            if save_patid is not None:
                save_4d_volume(new_4d_volume, save_patid, output_dir, file_suffix, save_sr_spacing, save_origin,
                               save_direction, myargs, save_original_spacing, resample=resample,
                               is_arvc_labels=is_arvc_labels, new_4d_labels=new_4d_labels,
                               output_dir_labels=output_dir_labels)
                if generate_inbetween_slices and verbose:
                        print("SSIM / PSRN / VIF: {:.3f} / {:.3f} / {:.3f}".format(ssim_results[-1], psnr_results[-1],
                                                                                   vif_results[-1]))

            new_4d_volume = np.empty((0, *list(new_images.shape)))
            new_4d_labels = None if labels is None else np.empty((0, *list(new_images.shape)))
        if save_volumes and is_4d:
            new_4d_volume = np.vstack((new_4d_volume, new_images[None]))
            new_4d_labels = None if labels is None else np.vstack((new_4d_labels, new_labels[None]))
        save_sr_spacing = sr_spacing[:]
        save_origin = origin
        save_direction = direction
        save_patid = pat_id
        save_original_spacing = original_spacing
        if generate_inbetween_slices:
            ssim_results, psnr_results, vif_results, lpips_results, \
                ssim_res_synth, psnr_res_synth, vif_res_synth, lpips_res_synth,\
                ssim_res_recon, psnr_res_recon, vif_res_recon, lpips_res_recon = \
                compute_metrics(images if image_hr is None else image_hr, new_images, downsample_steps,
                                ssim_results, psnr_results, vif_results, lpips_results,
                                ssim_res_synth, psnr_res_synth, vif_res_synth, lpips_res_synth,
                                ssim_res_recon, psnr_res_recon, vif_res_recon, lpips_res_recon,
                                compute_percept_loss=compute_percept_loss,
                                percept_loss=percept_loss,
                                normalize=normalize, eval_axis=eval_axis)

            if not is_4d and verbose:
                print("SSIM / PSRN / VIF: {:.3f} / {:.3f} / {:.3f}".format(ssim_results[-1], psnr_results[-1],
                                                                           vif_results[-1]))
    if is_4d:
        if save_volumes:
            save_4d_volume(new_4d_volume, save_patid, output_dir, file_suffix, save_sr_spacing, save_origin,
                           save_direction, myargs, save_original_spacing, resample=resample,
                           is_arvc_labels=is_arvc_labels, new_4d_labels=new_4d_labels,
                           output_dir_labels=output_dir_labels)

        if generate_inbetween_slices and verbose:
            print("SSIM / PSRN / VIF: {:.3f} / {:.3f} / {:.3f}".format(ssim_results[-1], psnr_results[-1],
                                                                           vif_results[-1]))
    if save_volumes and interpol_filter is None:
        if hasattr(trainer, 'model_file') and trainer.model_file is not None:
            model_nbr = trainer.model_file.split(os.sep)[-1].replace('.models', '')
            readme = os.path.join(output_dir, "README_{}_".format(model_nbr) + file_suffix + ".txt")
        else:
            readme = os.path.join(output_dir, "README_" + file_suffix + ".txt")
        Path(readme).touch()
    if generate_inbetween_slices:
        mean_ssim, std_ssim, mean_psnr, std_psnr, mean_vif, std_vif, mean_lpips, std_lpips = \
                compute_mean_metrics(ssim_results, psnr_results, vif_results, lpips_results)

        print("Total - SSIM / PSRN / VIF / LPIPS: {:.3f} ({:.2f}) / {:.2f} ({:.2f}) / "
              "{:.3f} ({:.2f}) / {:.3f} ({:.2f})".format(mean_ssim, std_ssim, mean_psnr, std_psnr,
                                                         mean_vif, std_vif, mean_lpips, std_lpips))
        if eval_axis == 0:
            mean_ssim_recon, std_ssim_recon, mean_psnr_recon, std_psnr_recon, mean_vif_recon, std_vif_recon, \
            mean_lpips_recon, std_lpips_recon = \
                compute_mean_metrics(ssim_res_recon, psnr_res_recon, vif_res_recon, lpips_res_recon)
            print("Reconstruction - SSIM / PSRN / VIF / LPIPS: {:.3f} ({:.2f}) / {:.2f} ({:.2f}) / "
                  "{:.3f} ({:.2f}) / {:.3f} ({:.2f})".format(mean_ssim_recon, std_ssim_recon,
                                                             mean_psnr_recon, std_psnr_recon,
                                                             mean_vif_recon, std_vif_recon, mean_lpips_recon,
                                                             std_lpips_recon))
            mean_ssim_synth, std_ssim_synth, mean_psnr_synth, std_psnr_synth, mean_vif_synth, std_vif_synth, \
            mean_lpips_synth, std_lpips_synth = \
                compute_mean_metrics(ssim_res_synth, psnr_res_synth, vif_res_synth, lpips_res_synth)
            print("Synthesis - SSIM / PSRN / VIF / LPIPS: {:.3f} ({:.2f}) / {:.2f} ({:.2f}) / "
                  "{:.3f} ({:.2f}) / {:.3f} ({:.2f})".format(mean_ssim_synth, std_ssim_synth,
                                                             mean_psnr_synth, std_psnr_synth,
                                                             mean_vif_synth, std_vif_synth, mean_lpips_synth,
                                                             std_lpips_synth))
        result_dict = {'ssim': ssim_results, 'psnr': psnr_results, 'vif': vif_results, 'lpips': lpips_results,
                'ssim_synth': ssim_res_synth, 'psnr_synth': psnr_res_synth, 'vif_synth': vif_res_synth,
                'lpips_synth': lpips_res_synth, 'ssim_recon': ssim_res_recon, 'psnr_recon': psnr_res_recon,
                'vif_recon': vif_res_recon, 'lpips_recon': lpips_res_recon}
        return result_dict
    else:
        return None, None, None
