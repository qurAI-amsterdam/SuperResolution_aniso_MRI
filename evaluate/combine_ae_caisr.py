import numpy as np
import os
import copy
from datasets.common import get_images_in_dir
from evaluate import generate_synth_slices_mask


def create_combined_images(dataset, downsample_steps, patid_list, ae_exper_id, caisr_exper_id,
                           do_save=False):

    exper_src_dir = os.path.expanduser("~/expers/sr/{}".format(dataset))
    file_suffix = "_ni0{}.nii.gz".format(downsample_steps - 1)
    # AE
    if not isinstance(ae_exper_id, dict):
        ae_path_method = os.path.join(exper_src_dir, "ae/{}/images_sr_ip".format(ae_exper_id))
        data_generator_ae = get_images_in_dir(ae_path_method, file_suffix=file_suffix,
                                              rescale_int=False,
                                              do_downsample=False,
                                              downsample_steps=None, patid_list=patid_list)
        print("Loaded from {} volumes from {} ({})".format(len(data_generator_ae), ae_path_method,
                                                           file_suffix))
    else:
        data_generator_ae = ae_exper_id
    # CAISR
    if not isinstance(caisr_exper_id, dict):
        caisr_path_method = os.path.join(exper_src_dir, "ae_combined/{}/images_sr_ip".format(caisr_exper_id))
        data_generator_caisr = get_images_in_dir(caisr_path_method, file_suffix=file_suffix,
                                                 rescale_int=False,
                                                 do_downsample=False,
                                                 downsample_steps=None, patid_list=patid_list)
        print("Loaded from {} volumes from {} ({})".format(len(data_generator_caisr), caisr_path_method,
                                                           file_suffix))
    else:
        data_generator_caisr = caisr_exper_id
    data_dict = {}
    for patid, image_dict in data_generator_ae.items():
        ae_images = image_dict['image']
        caisr_image = data_generator_caisr[patid]['image']
        num_slices = ae_images.shape[1] if ae_images.ndim == 4 else ae_images.shape[0]
        r_mask, _ = generate_synth_slices_mask(num_slices, downsample_steps)
        # print(ae_images.shape, caisr_image.shape, num_slices, downsample_steps)
        # print("1. ", len(r_mask), r_mask)
        if len(r_mask) != num_slices:
            # we need to extend with Trues for reconstructions
            l = list(r_mask)
            for i in range(num_slices - len(r_mask)):
                l.extend([True])
            r_mask = np.array(l)
            # print("2. ", len(r_mask), r_mask)
        new_image = copy.deepcopy(caisr_image)
        # copy reconstructions (+omitted slices due to downsampling) from ae generated volume
        if ae_images.ndim == 4:
            new_image[:, r_mask] = ae_images[:, r_mask]
        else:
            new_image[r_mask] = ae_images[r_mask]
        data_dict[patid] = {'image': new_image, 'patient_id': patid, 'spacing': image_dict['spacing']}
    return data_dict
