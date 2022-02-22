import os
import numpy as np
import SimpleITK as sitk
from glob import glob
from datasets.ACDC.data import rescale_intensities, apply_2d_zoom_3d

SPACING = tuple((8, 1.25, 1.25))


def get_all_images(root_dir='~/data/sunnybrook/sax', patid_list=None, resample=False,
                     rescale=False, new_spacing=None, limited_load=False, file_suffix='_ES.mhd',
                     as4d=False):
    if resample and new_spacing is None:
        raise ValueError("Error - if resample is True new_spacing must be specified!")
    search_mask = os.path.expanduser(root_dir) + os.sep + "*" + file_suffix
    file_list = glob(search_mask)
    if len(file_list) == 0:
        raise FileNotFoundError("Error - no files found for search mask {}".format(search_mask))
    if patid_list is not None:
        file_list = [fname for fname in file_list if os.path.basename(fname).replace(file_suffix, "") in patid_list]

    if limited_load:
        file_list = file_list[:20]
    for fname in file_list:
        img_es = sitk.GetArrayFromImage(sitk.ReadImage(fname))
        img_ed = sitk.GetArrayFromImage(sitk.ReadImage(fname.replace("ES", "ED")))
        img_es = np.flip(img_es, axis=2)
        img_ed = np.flip(img_ed, axis=2)
        # print("Patient id ", os.path.basename(fname).replace(file_suffix, ""))
        patient_id =os.path.basename(fname).replace(file_suffix, "")
        spacing = SPACING
        original_spacing = SPACING
        if resample:
            img_es = apply_2d_zoom_3d(img_es, spacing, do_blur=True, new_spacing=new_spacing)
            img_ed = apply_2d_zoom_3d(img_ed, spacing, do_blur=True, new_spacing=new_spacing)
            spacing = np.array([original_spacing[0], new_spacing[1], new_spacing[2]]).astype(np.float32)
        if rescale:
            img_es = rescale_intensities(img_es, percs=tuple((1, 99))).astype(np.float32)
            img_ed = rescale_intensities(img_ed, percs=tuple((1, 99))).astype(np.float32)
        if as4d:
            img4d = np.concatenate((img_ed[None], img_es[None]), axis=0)
            yield {'image': img4d, 'spacing': spacing, 'patient_id': patient_id,
                       'frame_id': 0, 'num_frames': 2,
                       'cardiac_phase': 'ED', 'structures': [], 'original_spacing': original_spacing}
        else:
            yield {'image': img_ed, 'spacing': spacing, 'patient_id': patient_id,
                   'frame_id': 0, 'num_frames':  2,
                    'cardiac_phase': 'ED', 'structures': [], 'original_spacing': original_spacing}
            yield {'image': img_es, 'spacing': spacing, 'patient_id': patient_id,
                   'frame_id': 1, 'num_frames': 2,
                   'cardiac_phase': 'ES', 'structures': [], 'original_spacing': original_spacing}


def get_all_images4d(root_dir='~/data/sunnybrook/sax', patid_list=None, resample=False,
                       rescale=False, new_spacing=None, limited_load=False, file_suffix='_ES.mhd'):
    if resample and new_spacing is None:
        raise ValueError("Error - if resample is True new_spacing must be specified!")
    search_mask = os.path.expanduser(root_dir) + os.sep + "*" + file_suffix
    file_list = glob(search_mask)
    if len(file_list) == 0:
        raise FileNotFoundError("Error - no files found for search mask {}".format(search_mask))
    if patid_list is not None:
        file_list = [fname for fname in file_list if os.path.basename(fname).replace(file_suffix, "") in patid_list]

    if limited_load:
        file_list = file_list[:20]
    image_dict = {}
    for fname in file_list:
        img_es = sitk.GetArrayFromImage(sitk.ReadImage(fname))
        img_ed = sitk.GetArrayFromImage(sitk.ReadImage(fname.replace("ES", "ED")))
        img_es = np.flip(img_es, axis=2)
        img_ed = np.flip(img_ed, axis=2)
        # print("Patient id ", os.path.basename(fname).replace(file_suffix, ""))
        patient_id = os.path.basename(fname).replace(file_suffix, "")
        spacing = SPACING
        original_spacing = SPACING
        if resample:
            img_es = apply_2d_zoom_3d(img_es, spacing, do_blur=True, new_spacing=new_spacing)
            img_ed = apply_2d_zoom_3d(img_ed, spacing, do_blur=True, new_spacing=new_spacing)
            spacing = np.array([original_spacing[0], new_spacing[1], new_spacing[2]]).astype(np.float32)
        if rescale:
            img_es = rescale_intensities(img_es, percs=tuple((1, 99))).astype(np.float32)
            img_ed = rescale_intensities(img_ed, percs=tuple((1, 99))).astype(np.float32)
        img4d = np.concatenate((img_ed[None], img_es[None]), axis=0)
        image_dict[patient_id] = {'image': img4d, 'spacing': spacing, 'patient_id': patient_id,
               'frame_id': 0, 'num_frames': 2,
               'cardiac_phase': 'ED', 'structures': [], 'original_spacing': original_spacing}

    return image_dict
