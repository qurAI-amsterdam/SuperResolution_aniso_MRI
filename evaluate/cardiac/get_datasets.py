import numpy as np
import os
from datasets.ACDC.data import acdc_all_image4d
from datasets.ARVC.dataset import arvc_get_evaluate_set
from datasets.ACDC.data_with_labels import get_4d_edes_image_generator, get_4d_image_generator
from datasets.ARVC.sr_data_with_labels import get_arvc_4d_edes_image_generator
from datasets.sunnybrook.dataset import get_all_images as get_all_images_sunnyb
from datasets.sax_dataset import get_sax_images_gen
from datasets.data_config import get_config


def get_cardiac_dataset_generator(dataset_name, patid_list=None):
    dtaset_config = get_config(dataset_name)
    if dataset_name == "ACDC":

        print("INFO - evaluating on {} dataset with {} patients".format(dataset_name, len(patid_list)))
        data_generator = acdc_all_image4d(root_dir=os.path.expanduser(dtaset_config.short_axis_dir), resample=True,
                                          rescale=True, new_spacing=tuple((1, 1.4, 1.4)),
                                          limited_load=False, patid_list=patid_list)
    elif dataset_name == 'ACDCLBL':
        print("INFO - evaluating on {} dataset with {} patients".format(dataset_name, len(patid_list)))
        data_generator = get_4d_edes_image_generator(root_dir=os.path.expanduser(dtaset_config.short_axis_dir),
                                                     rescale=True, resample=True, limited_load=False,
                                                     new_spacing=tuple((1, 1.4, 1.4)),
                                                     pat_nums=patid_list)
    elif dataset_name == 'ACDC4DLBL':
        print("INFO - evaluating on {} dataset with {} patients".format(dataset_name, len(patid_list)))
        data_generator = get_4d_image_generator(dtaset_config.short_axis_dir, dataset=None, rescale=True, resample=True,
                                                limited_load=False, new_spacing=tuple((1, 1.4, 1.4)),
                                                rs=np.random.RandomState(1234), pat_nums=patid_list)
        resample = True
    elif dataset_name == "ARVC":
        # IMPORTANT: OVERWRITE resampling to original spacing because we always leave original in-plane resolution
        # for ARVC (16-6-2021)
        resample = False
        type_of_set = "validation"
        print("INFO - evaluating on {} dataset for {} volumes: resample={}".format(dataset_name, type_of_set,
                                                                                   resample))
        # 16-6-2021: disabled resampling to ACDC 1.4 in-plane resolution.
        data_generator = arvc_get_evaluate_set(type_of_set, limited_load=False, resample=resample, rescale=True,
                                               patid=None,
                                               all_frames=True,
                                               new_spacing=None)

    elif dataset_name == 'ARVCLBL':
        print("INFO - evaluating on {} dataset for test volumes".format(dataset_name))
        data_generator = get_arvc_4d_edes_image_generator("validation", limited_load=False, resample=True, rescale=True,
                                                          patid=None,
                                                          new_spacing=np.array([1, 1.4, 1.4]))

    elif dataset_name == "sunnybrook":
        print("INFO - evaluating on {} dataset for test volumes".format(dataset_name))
        data_generator = get_all_images_sunnyb(root_dir='~/data/sunnybrook/sax', patid_list=None, resample=True,
                                                rescale=True, new_spacing=np.array([1, 1.4, 1.4]), limited_load=False,
                                                file_suffix='_ES.mhd',
                                                as4d=False)
    elif dataset_name == "PIE":
        print("INFO - evaluating on {} dataset: {}".format(dataset_name, dtaset_config.short_axis_dir))
        patid_list = ['00251']  # , '00252', '00253']
        data_generator = get_sax_images_gen(dataset_name, patid_list, rescale=True, resample=False,
                                            new_spacing=None)  # np.array([1, 1.4, 1.4])
    else:
        raise ValueError("Error - unknown dataset name {} is not supported".format(dataset_name))

    return data_generator
