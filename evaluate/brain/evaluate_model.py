import os
from kwatsch.get_trainer import get_trainer_dynamic
from evaluate.create_HR_images import create_hr_images, save_metrics_to_file
from datasets.data_config import get_config
from datasets.common_brains import get_images
from datasets.dHCP.create_dataset import get_patient_ids
from datasets.OASIS.dataset import get_oasis_patient_ids
from datasets.ADNI.dataset import get_patient_ids as get_patient_ids_adni
from datasets.MNIST.data3d import get_mnist_ids
from torchvision import transforms
from datasets.shared_transforms import AdjustToPatchSize, CenterCrop


def get_test_dataset(dataset, patient_id=None, downsample=False, downsample_steps=None,
                     type_of_set="test", patch_size=None, include_hr_images=False, patid_list=None):
    transform = None
    data_config = get_config(dataset)
    if dataset == "dHCP":
        if patid_list is None:
           patid_list = get_patient_ids(type_of_set, data_config.image_dir)
        data_generator = get_images(patid_list, dataset, rescale_int=True, do_downsample=downsample,
                                    downsample_steps=downsample_steps,
                                    include_hr_images=include_hr_images, limited_load=False, verbose=True)
    elif dataset == "ADNI":
        if patid_list is None:
            patid_list = get_patient_ids_adni(type_of_set, data_config.image_dir)
        transform = transforms.Compose([AdjustToPatchSize(tuple((patch_size, patch_size))),
                                        CenterCrop(patch_size)])
        data_generator = get_images(patid_list, dataset, rescale_int=True, do_downsample=downsample,
                                    downsample_steps=downsample_steps,
                                    include_hr_images=include_hr_images, limited_load=False, verbose=True)
    elif dataset == "OASIS":
        assert patch_size is not None
        transform = transforms.Compose([AdjustToPatchSize(tuple((patch_size, patch_size))),
                                        CenterCrop(patch_size)])
        if patid_list is None:
            patid_list = get_oasis_patient_ids(type_of_set)
        data_generator = get_images(patid_list, dataset, rescale_int=True, do_downsample=downsample,
                                    downsample_steps=downsample_steps,
                                    include_hr_images=include_hr_images, limited_load=False)
    elif dataset in ["MNIST3D", 'MNIST', 'MNISTRoto']:
        assert patch_size is not None
        if patid_list is None:
            patid_list = get_mnist_ids(type_of_set)
        # Important: for MNISTRoto we use same synthetic volumes as for MNIST3D
        dataset = 'MNIST3D' if dataset == 'MNISTRoto' else dataset
        if patch_size is not None:
            transform = transforms.Compose([AdjustToPatchSize(tuple((patch_size, patch_size)))])
        data_generator = get_images(patid_list, dataset, rescale_int=False, do_downsample=downsample,
                                    downsample_steps=downsample_steps,
                                    include_hr_images=include_hr_images, limited_load=False)
    else:
        raise ValueError("Error - unsupported dataset {}".format(dataset))

    return data_generator, transform


def create_brain_volumes(exper_src_path, model_nbr, pat_list=None, eval_axis=0, model_nbr_sr=None,
                         eval_patch_size=256, save_volumes=False, downsample_steps=3, compute_percept_loss=True,
                         use_original_slice=False, generate_inbetween_slices=True, num_interpolations=None,
                         output_dir=None, do_save_metrics=False):
    exper_src_path = os.path.expanduser(exper_src_path)
    trainer, myargs = get_trainer_dynamic(src_path=exper_src_path, model_nbr=model_nbr, model_nbr_sr=model_nbr_sr)
    data_generator, transform = get_test_dataset(myargs['dataset'], patid_list=pat_list, type_of_set="test",
                                                 patch_size=eval_patch_size, downsample_steps=downsample_steps,
                                                 include_hr_images=True)
    base_out_dir = "images_sr"  # "images_centered_sr"    images_noncentered
    print("WARNING - Current model-nbr >>> {} <<< "
          "train patch-size/test patch-size {}/{}".format(model_nbr, myargs['width'],
                                                          eval_patch_size))
    print("INFO - evaluating on {} patients".format(len(data_generator)))
    if generate_inbetween_slices and myargs['downsample_steps'] != downsample_steps:
        if myargs['dataset'] not in ['MNIST3D', 'MNISTRoto']:
            raise ValueError("Error - generate_inbetween_slices is {} but downsample_steps {}"
                             " does not match parameter {} used during training!".format(generate_inbetween_slices,
                                                                                         myargs['downsample_steps'],
                                                                                         downsample_steps))
        else:
            print("WARNING !!! downsample_steps {}"
                  " does not match parameter {} used during training!".format(myargs['downsample_steps'],
                                                                                         downsample_steps))

    normalize = False  # normalization before computing evaluation metrics (ssim, psnr, etc)
    if generate_inbetween_slices and num_interpolations is None:
        num_interpolations = downsample_steps - 1
    if use_original_slice:
        file_suffix = "ni{:02}_oslices".format(num_interpolations)
    else:
        file_suffix = "ni{:02}".format(num_interpolations)
    # !!! IMPORTANT Slightly tricky, I know. But, when generating volumes not used for quantitative evaluation,
    #     e.g. num_interpolation = 6 for dHCP, we DOWNSAMPLED the HR dHCP dataset with 3 (0.5 to 1.5mmm) but
    #     we have to set downsample_steps to None before calling create_hr_images because otherwise volumes will
    #     be downsampled again
    if not generate_inbetween_slices:
        print("!!! Warning !!! Generation {} new slices and mimick LR volumes with downsampling "
              "factor {}".format(num_interpolations, downsample_steps))
    is_4d = False
    print(myargs)
    result_metrics_dict = create_hr_images(data_generator, myargs, trainer,
                                                   num_interpolations=num_interpolations,
                                                   downsample_steps=downsample_steps,
                                                   use_original_slice=use_original_slice,
                                                   is_4d=is_4d, transform=transform, normalize=normalize,
                                                   generate_inbetween_slices=generate_inbetween_slices,
                                                    file_suffix=file_suffix,
                                                   save_volumes=save_volumes,
                                                   compute_percept_loss=compute_percept_loss, verbose=True,
                                                   base_out_dir=base_out_dir, eval_axis=eval_axis,
                                           output_dir=output_dir)

    if do_save_metrics:
        # save_metrics(os.path.join(exper_src_path, 'results'), myargs['dataset'], result_metrics_dict,
        # downsample_steps, 'model{:04d}'.format(model_nbr), eval_axis)
        metrics_dir = os.path.join(exper_src_path, 'results')
        if not os.path.isdir(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=False)
        fname = os.path.join(metrics_dir, 'metrics_m{}_{}x_axis0.npz'.format(model_nbr, num_interpolations + 1))
        save_metrics_to_file(result_metrics_dict, fname)
    return result_metrics_dict


def create_brain_volumes_conventional_interpolation(dataset, interpol_filter, expand_factor=None,
                                                   eval_patch_size=256, save_volumes=False, downsample_steps=3,
                                                   compute_percept_loss=True,
                                                   use_original_slice=False, generate_inbetween_slices=True,
                                                   num_interpolations=None,
                                                   output_dir=None, do_save_metrics=False):
    if output_dir is None:
        output_dir = os.path.expanduser('~/expers/sr/{}/conventional'.format(dataset))
    else:
        output_dir = os.path.expanduser(output_dir)
    assert dataset in ['ADNI', 'dHCP', 'OASIS']
    data_generator, transform = get_test_dataset(dataset, type_of_set="test", patch_size=eval_patch_size,
                                                 downsample_steps=downsample_steps, include_hr_images=True)

    patient_id = None
    normalize = False  # normalization before computing evaluation metrics (ssim, psnr, etc)
    if generate_inbetween_slices and expand_factor is None:
        expand_factor = downsample_steps
    if num_interpolations is None:
        num_interpolations = downsample_steps - 1

    file_suffix = "ni{:02}".format(num_interpolations)
    is_4d = False
    dummy_args = {'model': interpol_filter, 'output_dir': output_dir}
    print("WARNING - Creating volumes with conventional interpolation method"
          " {} - eval-patch-size {} - expand factor {} - "
          "downsample steps {} ".format(interpol_filter, eval_patch_size, expand_factor, downsample_steps))
    print("INFO - evaluating on {} patients".format(len(data_generator)))
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
                                           base_out_dir=None)
    if do_save_metrics:
        # save_metrics(os.path.join(exper_src_path, 'results'), myargs['dataset'], result_metrics_dict,
        # downsample_steps, 'model{:04d}'.format(model_nbr), eval_axis)
        metrics_dir = os.path.join(output_dir, 'results')
        if not os.path.isdir(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=False)
        fname = os.path.join(metrics_dir, 'metrics_{}_{}x_axis0.npz'.format(interpol_filter, downsample_steps))
        save_metrics_to_file(result_metrics_dict, fname)

    return result_metrics_dict