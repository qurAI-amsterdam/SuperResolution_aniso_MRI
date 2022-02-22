import os
import numpy as np
from torchvision import transforms
from datasets.shared_transforms import CenterCrop, AdjustToPatchSize
from kwatsch.get_trainer import get_trainer_dynamic
from evaluate.create_HR_images import create_hr_images, save_metrics_to_file
from datasets.ACDC.data import acdc_all_image4d
from datasets.ARVC.dataset import arvc_get_evaluate_set
from datasets.data_config import get_config
from datasets.sunnybrook.dataset import get_all_images as get_all_images_sunnyb
from evaluate.cardiac.get_datasets import get_cardiac_dataset_generator


def create_acdc_volumes(exper_src_path, model_nbr, patid_list, dataset_name="ACDC", eval_axis=0,
                                eval_patch_size=128, save_volumes=False, downsample_steps=2, compute_percept_loss=True,
                                use_original_slice=False, generate_inbetween_slices=True, num_interpolations=None,
                        interpol_filter=None, output_dir=None, model_nbr_sr=None, resample=False, save_metrics=False):

    exper_src_path = os.path.expanduser(exper_src_path)
    data_generator = get_cardiac_dataset_generator(dataset_name, patid_list=patid_list)
    # trainer, myargs = get_trainer(src_path=exper_src_path, model_nbr=model_nbr)
    trainer, myargs = get_trainer_dynamic(src_path=exper_src_path, model_nbr=model_nbr, model_nbr_sr=model_nbr_sr)
    base_out_dir = "images_sr"  # "images_centered_sr"    images_noncentered
    print("WARNING - {} - Current model-nbr >>> {} <<< "
          "train patch-size/test patch-size {}/{}".format(dataset_name, model_nbr, myargs['width'],
                                                          eval_patch_size))

    if dataset_name in ['ARVC', 'PIE']:
        print("INFO - Using no TRANSFORM. Resample back to original spacing: {}".format(resample))
        transform = None
    else:
        transform = transforms.Compose([AdjustToPatchSize(tuple((eval_patch_size, eval_patch_size))),
                                        CenterCrop(eval_patch_size)])
    patient_id = None
    normalize = False  # normalization before computing evaluation metrics (ssim, psnr, etc)
    if generate_inbetween_slices and num_interpolations is None:
        num_interpolations = downsample_steps - 1
    if num_interpolations > 1:
        # this property is/was only relevant for the alpha ae, that also predicted the mixing coefficients.
        # this is no longer in use, but for now we leave it like this. So don't worry!
        trainer.eval_fixed_coeff = True
    if use_original_slice:
        file_suffix = "ni{:02}_oslices".format(num_interpolations)
    else:
        file_suffix = "ni{:02}".format(num_interpolations)
    is_4d = True

    print(myargs)
    result_metrics_dict = create_hr_images(data_generator, myargs, trainer,
                                                   num_interpolations=num_interpolations,
                                                   downsample_steps=downsample_steps,
                                                   use_original_slice=use_original_slice,
                                                   is_4d=is_4d, transform=transform, normalize=normalize,
                                                   generate_inbetween_slices=generate_inbetween_slices,
                                                   patient_id=patient_id, file_suffix=file_suffix,
                                                   save_volumes=save_volumes,
                                                   compute_percept_loss=compute_percept_loss, verbose=True,
                                                   output_dir=output_dir,
                                                   base_out_dir=base_out_dir, eval_axis=eval_axis,
                                                   is_arvc_labels=True if dataset_name == 'ARVCLBL' else False,
                                           resample=resample)
    if save_metrics:
        metrics_dir = os.path.join(exper_src_path, 'results')
        if not os.path.isdir(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=False)
        fname = os.path.join(metrics_dir, 'metrics_m{}_{}x_axis0.npz'.format(model_nbr, num_interpolations+1))
        save_metrics_to_file(result_metrics_dict, fname)
    return result_metrics_dict


def create_acdc_volumes_conventional_interpolation(interpol_filter, patid_list, expand_factor=None,
                                                   dataset_name="ACDC", eval_axis=0,
                                                   eval_patch_size=128, save_volumes=False, downsample_steps=2,
                                                   compute_percept_loss=True,
                                                   use_original_slice=False, generate_inbetween_slices=True,
                                                   num_interpolations=None,
                                                   output_dir='~/expers/sr/ACDC/conventional'):
    output_dir = os.path.expanduser(output_dir)
    dtaset_config = get_config(dataset_name)
    if dataset_name == 'ACDC':
        data_generator = acdc_all_image4d(root_dir=os.path.expanduser(dtaset_config.short_axis_dir), resample=True,
                                          rescale=True, new_spacing=tuple((1, 1.4, 1.4)),
                                          limited_load=False, patid_list=patid_list)
    elif dataset_name == 'ARVC':
        print("INFO - evaluating on {} dataset for test volumes".format(dataset_name))
        data_generator = arvc_get_evaluate_set("test", limited_load=False, resample=True, rescale=True,
                                               patid=None,
                                               all_frames=True,
                                               new_spacing=np.array([1, 1.4, 1.4]))
    elif dataset_name == "sunnybrook":
        print("INFO - evaluating on {} dataset for test volumes".format(dataset_name))
        data_generator = get_all_images_sunnyb(root_dir='~/data/sunnybrook/sax', patid_list=None, resample=True,
                                                rescale=True, new_spacing=np.array([1, 1.4, 1.4]), limited_load=False,
                                                file_suffix='_ES.mhd',
                                                as4d=False)
    else:
        raise ValueError("Error - unknown dataset name {} is not supported".format(dataset_name))
    base_out_dir = "images_sr"  # "images_centered_sr"    images_noncentered
    transform = transforms.Compose([AdjustToPatchSize(tuple((eval_patch_size, eval_patch_size))),
                                    CenterCrop(eval_patch_size)])
    patient_id = None
    normalize = False  # normalization before computing evaluation metrics (ssim, psnr, etc)
    if generate_inbetween_slices and expand_factor is None:
        expand_factor = downsample_steps
    if generate_inbetween_slices and num_interpolations is None:
        num_interpolations = downsample_steps - 1
    else:
        # we're generating volumes with expand_factor higher than 2 (not for test purposes)
        assert expand_factor is not None
        num_interpolations = expand_factor - 1

    file_suffix = "ni{:02}".format(num_interpolations)
    is_4d = True
    dummy_args = {'model': interpol_filter, 'output_dir': output_dir}
    print("WARNING - Creating volumes with conventional interpolation method"
          " {} - eval-patch-size {} - expand factor {} - "
          "downsample steps {} ".format(interpol_filter, eval_patch_size, expand_factor, downsample_steps))
    if patid_list is not None:
        print("INFO - evaluating on {} patients".format(len(patid_list)))
    result_metrics_dict = create_hr_images(data_generator, dummy_args, interpol_filter=interpol_filter,
                                           num_interpolations=num_interpolations,
                                           expand_factor=expand_factor,
                                           downsample_steps=downsample_steps,
                                           use_original_slice=use_original_slice,
                                           is_4d=is_4d, transform=transform, normalize=normalize,
                                           generate_inbetween_slices=generate_inbetween_slices,
                                           patient_id=patient_id, file_suffix=file_suffix,
                                           save_volumes=save_volumes,
                                           compute_percept_loss=compute_percept_loss, verbose=True,
                                           base_out_dir=base_out_dir,
                                           eval_axis=eval_axis)
    return result_metrics_dict