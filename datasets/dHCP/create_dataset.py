import numpy as np
import SimpleITK as sitk
from pathlib import Path
import os
import yaml
from tqdm import tqdm

from datasets.ACDC.data import sitk_save
from nilearn.masking import compute_background_mask
import nibabel as nib
from datasets.common import rescale_intensities


def determine_max_slice_brain_stem(img_ref, axis=1, cls=5):
    mask = img_ref == cls
    nzero = np.nonzero(mask)
    return np.min(nzero[axis]), np.max(nzero[axis])


def get_image_with_background_mask(file_obj, ):

    myimage = nib.load(str(file_obj.resolve()))
    spacing = np.array(myimage.header.get_zooms()[::-1]).astype(np.float64)
    mask = compute_background_mask(myimage)
    np_image = myimage.get_data().T
    np_mask = mask.get_data().T
    return {'image': np_image, 'mask': np_mask, 'spacing': spacing}


def get_images(dataset, src_path="~/data/dHCP/", rescale_int=False, int_perc=tuple((0, 100)), limited_load=False,
               do_downsample=False, downsample_steps=3, patid=None, file_suffix=".nii.gz"):

    src_path = os.path.expanduser(src_path)
    filepath_generator = Path(src_path).rglob('*' + file_suffix)
    if dataset is not None:
        dataset = dataset.lower()
        patient_ids = get_patient_ids(dataset, src_path)
        filepath_list = [file_obj for file_obj in filepath_generator if int(file_obj.name.replace(file_suffix, "")) in patient_ids]
    else:
        # we assume file basenames are "120230.nii.gz" but actually sometimes we append interpolation method after
        # the patient_id e.g. "120230_acai.nii.gz". In that case we need to strip "_acai"
        filepath_list = [file_obj for file_obj in filepath_generator]
    if len(filepath_list) == 0:
        raise ValueError("ERROR - get_images - no files found in {}".format(src_path))
    filepath_list.sort()
    if limited_load:
        filepath_list = filepath_list[:4]
    image_dict = {}
    for file_obj in tqdm(filepath_list, desc="Loading {} volumes from {}".format(len(filepath_list), src_path)):
        patient_id = int(file_obj.name.replace(file_suffix, "").split("_")[0])
        if patid is not None:
            if patient_id != patid:
                continue
        img = sitk.ReadImage(str(file_obj.resolve()))
        np_image = sitk.GetArrayFromImage(img).astype(np.float32)
        if do_downsample:
            np_image = np_image[::int(downsample_steps)]
        if rescale_int:
            np_image = rescale_intensities(np_image, percs=int_perc)

        image_dict[patient_id] = {'image': np_image,
                             'spacing': np.array(img.GetSpacing()[::-1]).astype(np.float64),
                             'origin': img.GetOrigin(), 'direction': img.GetDirection(),
                             'patient_id': patient_id, "num_slices": np_image.shape[0]}
    return image_dict


def pad_image(image, patch_size):
    _, w, h = image.shape
    delta_w_l, delta_w_r, delta_h_l, delta_h_r = 0, 0, 0, 0
    if w < patch_size[0]:
        delta_w = patch_size[0] - w
        delta_w_l = delta_w // 2
        delta_w_r = delta_w // 2 if delta_w % 2 == 0 else int(delta_w_l + 1)
    if h < patch_size[1]:
        delta_h = patch_size[0] - h
        delta_h_l = delta_h // 2
        delta_h_r = delta_h // 2 if delta_h % 2 == 0 else int(delta_h_l + 1)
    image = np.pad(image,
                   ((0, 0), (delta_w_l, delta_w_r),
                    (delta_h_l, delta_h_r)),
                   'constant',
                   constant_values=(0,)).astype(np.float32)
    return image


def create_dataset(src_path="~/data/dHCP/", out_path="~/data/dHCP_cropped_256",
                   do_create_split_file=False, patient_id=None, patch_size=None):
    if patch_size is not None:
        if not isinstance(patch_size, tuple):
            patch_size = tuple((patch_size, patch_size))
    src_path = os.path.expanduser(src_path)
    out_path = os.path.expanduser(out_path)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    filepath_generator = Path(src_path).rglob('*.nii.gz')
    all_patids = []
    image_dict = {}

    for file_obj in filepath_generator:
        patid = int(file_obj.parts[5])
        if patient_id is not None:
            if patid != patient_id:
                continue
        data_dict = get_image_with_background_mask(file_obj)
        min_brain_ax1, max_brain_ax1 = determine_max_slice_brain_stem(data_dict['mask'], axis=1, cls=1)
        min_brain_ax2, max_brain_ax2 = determine_max_slice_brain_stem(data_dict['mask'], axis=2, cls=1)
        min_brain_ax0, max_brain_ax0 = determine_max_slice_brain_stem(data_dict['mask'], axis=0, cls=1)
        if patch_size is not None:
            if (max_brain_ax1 - min_brain_ax1) > patch_size[0]:
                too_long = (max_brain_ax1 - min_brain_ax1) - patch_size[0]
                print("WARNING - have to shorten y-direction of image by {}".format(too_long))
                min_brain_ax1 += 2
                max_brain_ax1 = max_brain_ax1 - (too_long - 2)
        np_cropped_img = data_dict['image'][min_brain_ax0:max_brain_ax0, min_brain_ax1:max_brain_ax1,
                                    min_brain_ax2:max_brain_ax2]
        np_cropped_mask = data_dict['mask'][min_brain_ax0:max_brain_ax0, min_brain_ax1:max_brain_ax1,
                                    min_brain_ax2:max_brain_ax2]

        new_image = np.zeros_like(np_cropped_mask).astype(np.float32)
        new_image[np_cropped_mask == 1] = np_cropped_img[np_cropped_mask == 1]
        if patch_size is not None:
            new_image = pad_image(new_image, patch_size)
        # Compared with standard (?) brain MR images (i compare with what i see in MeVisLab) the volumes
        # are 180 degrees rotated
        # new_image = np.rot90(new_image, 2, (1, 2))
        # new_image = np.rot90(new_image, 2, (0, 2))

        all_patids.append(patid)
        abs_filename = os.path.join(out_path, "{:06d}".format(patid) + ".nii.gz")
        sitk_save(abs_filename, new_image, spacing=data_dict['spacing'], dtype=np.float32, normalize=False)
        image_dict[patid] = {'image': new_image, 'spacing': data_dict['spacing']}
        print("INFO - Save to {}".format(abs_filename))
    if do_create_split_file:
        create_split_file(out_path, patid_list=all_patids)
    return image_dict


def create_split_file(out_path, patid_list=None, num_split=(200, 20, 20), rs=np.random.RandomState(1234)):
    # num_split: indicates split in absolute numbers between training, test and validation sets
    def numpy_array_to_native_python(np_arr) -> list:
        return [val.item() for val in np_arr]

    out_path = os.path.expanduser(out_path)
    if patid_list is None:
        filepath_generator = Path(out_path).rglob('*_t2w.nii.gz')
        patid_list = [int(path_obj.name.replace("_t2w.nii.gz", "")) for path_obj in filepath_generator]

    if len(patid_list) == 0:
        raise ValueError("ERROR - create split file - no files found in {}".format(out_path))

    ids = rs.permutation(np.array(patid_list))
    # create two lists of files
    train_offset = int(num_split[0])
    patids_train = numpy_array_to_native_python(ids[:train_offset])
    patids_test = numpy_array_to_native_python(ids[train_offset:train_offset + int(num_split[1])])
    patids_val = numpy_array_to_native_python(ids[train_offset + int(num_split[1]):train_offset + int(num_split[1]) + int(num_split[2])])
    split_config = {'training': patids_train,
                    'validation': patids_val,
                    'test': patids_test}
    output_file = os.path.join(out_path, "train_test_split.yaml")
    print("INFO - Saved split file to {}".format(output_file))
    with open(output_file, 'w') as fp:
        yaml.dump(split_config, fp)

    return split_config


def get_patient_ids(dataset, src_path="~/data/dHCP_cropped/"):
    src_path = os.path.expanduser(src_path)
    split_file = os.path.join(src_path, "train_test_split.yaml")
    if os.path.isfile(split_file):
        # load existing splits
        with open(split_file, 'r') as fp:
            patient_ids = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        print("Warning - load_split_file - creating NEW train/test split for cHCP dataset")
        patient_ids = create_split_file(src_path)
    return patient_ids[dataset]
