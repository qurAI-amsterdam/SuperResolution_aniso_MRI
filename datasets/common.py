import os
import yaml
import glob
import copy
import scipy.ndimage
import torch
import numpy as np
import SimpleITK as sitk
from torchvision import datasets, transforms
from torch.utils.data.sampler import RandomSampler
from datasets.data_config import get_config
from pathlib import Path
from torch.utils.data.sampler import Sampler


class MyRandomSampler(Sampler):
    def __init__(self, data_source, rs=np.random.RandomState(1234)):
        self.data_source = data_source
        self.rs = rs

    def set_seed(self, rs):
        self.rs = rs

    def __iter__(self):
        n = len(self.data_source)
        indexes = list(range(n))
        self.rs.shuffle(indexes)
        return iter(indexes)

    def __len__(self):
        return len(self.data_source)


def get_random_adjacent_slice(slice_id, num_slices, rs=np.random.RandomState(1234), step=1):
    num_slices -= 1
    if slice_id + step > num_slices:
        return slice_id - step
    elif slice_id == 0:
        return step
    elif slice_id - step < 0:
        return slice_id + step
    else:
        return rs.choice([slice_id - step, slice_id + step])


def get_paired_frames(num_frames, rs):
    c = rs.randint(2, size=1)[0]
    if c == 0:
        frames_from = np.repeat(np.array([0]), num_frames)
        frames_to = np.append(np.arange(1, num_frames), np.array([num_frames-1]))
    else:
        frames_from = np.repeat(np.array([num_frames-1]), num_frames)
        frames_to = np.append(np.arange(0, num_frames-1), np.array([0]))

    return frames_from, frames_to


def get_paired_slices(num_slices):
    joker_slice_id = int(np.random.randint(num_slices, size=1)[0])
    slices_1 = np.append(np.arange(0, num_slices - 1), np.array([joker_slice_id]))
    slices_2 = np.append(np.arange(1, num_slices), np.array([joker_slice_id]))
    c = np.random.randint(2, size=1)[0]
    if c == 0:
        slices_from = slices_1
        slices_to = slices_2
    else:
        slices_from = slices_2
        slices_to = slices_1
    return slices_from, slices_to


def get_leave_one_out_slices(num_slices):
    joker_slice_id1 = int(np.random.randint(1, num_slices - 1, size=1)[0])
    joker_slice_id2 = int(np.random.randint(1, num_slices - 1, size=1)[0])
    slices_1 = np.append(np.arange(num_slices - 2), np.array([joker_slice_id1 - 1, joker_slice_id2 - 1]))
    slices_2 = np.append(np.arange(2, num_slices), np.array([joker_slice_id1 + 1, joker_slice_id2 + 1]))
    c = np.random.randint(2, size=1)[0]
    if c % 2 == 0:
        slices_from = slices_1
        slices_to = slices_2
    else:
        slices_from = slices_2
        slices_to = slices_1

    targetid_interp = np.append(np.arange(2, num_slices) - 1, np.array([joker_slice_id1, joker_slice_id2]))
    return slices_from, slices_to, targetid_interp


class RandomTranslation(object):
    def __init__(self, patch_size=64, rs=np.random):
        self.rs = rs
        self.patch_size = patch_size

    def __call__(self, *args):
        new_image = np.zeros((self.patch_size, self.patch_size)).astype(np.uint8)
        image = args[0]
        image = np.array(image).astype(np.uint8)
        w, h = image.shape
        w_range, h_range = np.arange(0, self.patch_size - w), np.arange(0, self.patch_size - h)
        w_start, h_start = self.rs.choice(w_range), self.rs.choice(h_range)
        new_image[w_start:w_start + w, h_start:h_start + h] = image
        return new_image


def get_data_loaders_mnist(batch_size, use_cuda=True, data_dir='~/data/', test_batch_size=16,
                           transform_tr=transforms.Compose([transforms.ToTensor()]),
                           transform_te=None):
    if transform_te is None:
        transform_te = transform_tr
    data_dir = os.path.expanduser(data_dir)
    kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}
    training_set = datasets.MNIST(data_dir, train=True, download=True,
                       transform=transform_tr)
    train_loader = torch.utils.data.DataLoader(training_set,
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    test_set = datasets.MNIST(data_dir, train=False, transform=transform_te)
    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs)

    return train_loader, training_set, test_loader, test_set


def get_mnist_specific_test_batch(test_set, rs):

    a, b = [], []
    labels = {}
    iters = np.arange(0, len(test_set))
    rs.shuffle(iters)
    iters = list(iters)
    for i in iters:
        img_a, label_a = test_set[i]
        reduced_iters = copy.deepcopy(iters)
        reduced_iters.remove(i)
        if label_a not in labels.keys():
            for ii in reduced_iters:
                img_b, label_b = test_set[ii]
                if label_a == label_b:
                    a.append(img_a)
                    b.append(img_b)
                    labels[label_a] = None
                    break
        if len(labels) == 10:
            break
    a = np.concatenate(a, axis=0)
    b = np.concatenate(b, axis=0)
    return np.concatenate((a, b), axis=0)


def get_image_file_list(search_mask_img) -> list:
    files_to_load = glob.glob(search_mask_img)
    files_to_load.sort()
    if len(files_to_load) == 0:
        raise ValueError("ERROR - get_image_file_list - Can't find any files to load in {}".format(search_mask_img))
    return files_to_load


def apply_2d_zoom_4d(arr4d, spacing, new_spacing, order=1, do_blur=True, as_type=np.float32):
    """

        :param arr4d: [#timepoints, #slices, IH, IW]
        :param spacing: spacing has shape [#slices, IH, IW]
        :param new_vox_size: tuple(x, y)
        :param order: of B-spline
        :param do_blur: boolean (see below)
        :param as_type: return type of np array. We use this to indicate binary/integer labels which we'll round off
                        to prevent artifacts
        :return:
        """
    new_img4d = None
    for t in np.arange(arr4d.shape[0]):
        arr3d = arr4d[t]
        arr3d = apply_2d_zoom_3d(arr3d, spacing, new_spacing, order=order, do_blur=do_blur, as_type=as_type)
        new_img4d = np.vstack((new_img4d, np.expand_dims(arr3d, axis=0))) if new_img4d is not None else np.expand_dims(arr3d, axis=0)

    return new_img4d


def apply_2d_zoom_3d(arr3d, spacing, new_spacing, order=1, do_blur=True, as_type=np.float32):
    """

    :param arr3d: [#slices, IH, IW]
    :param spacing: spacing has shape [#slices, IH, IW]
    :param new_vox_size: tuple(x, y)
    :param order: of B-spline
    :param do_blur: boolean (see below)
    :param as_type: return type of np array. We use this to indicate binary/integer labels which we'll round off
                    to prevent artifacts
    :return:
    """
    if len(spacing) > 2:
        spacing = spacing[int(len(spacing) - 2):]

    if len(new_spacing) > 2:
        new_spacing = new_spacing[int(len(new_spacing) - 2):]

    zoom = np.array(spacing, np.float64) / new_spacing
    if do_blur:
        for z in range(arr3d.shape[0]):
            sigma = .25 / zoom
            arr3d[z, :, :] = scipy.ndimage.gaussian_filter(arr3d[z, :, :], sigma)

    resized_img = scipy.ndimage.interpolation.zoom(arr3d, tuple((1,)) + tuple(zoom), order=order)
    if as_type == np.int:
        # binary/integer labels
        resized_img = np.round(resized_img).astype(as_type)
    return resized_img


def read_nifty(fname, get_extra_info=False):
    img = sitk.ReadImage(fname)
    spacing = img.GetSpacing()[::-1]
    arr = sitk.GetArrayFromImage(img)
    if get_extra_info:
        return arr, spacing, img.GetDirection(), img.GetOrigin()
    return arr, spacing


def sitk_save(fname: str, arr: np.ndarray, spacing=None, dtype=np.float32, direction=None, origin=None):
    # IMPORTANT ASSUMPTIONS:
    # spacing: [z, y, x] will be flipped before save (see below)
    # direction: original sitk format (tuple)
    # origin: original sitk format (tuple -> xyz)
    if type(spacing) == type(None):
        spacing = np.ones((len(arr.shape),))

    if arr.ndim == 4:
        # 4d array
        if len(spacing) == 3:
            spacing = np.array([1, spacing[0], spacing[1], spacing[2]]).astype(np.float64)
        volumes = [sitk.GetImageFromArray(arr[v].astype(dtype), False) for v in range(arr.shape[0])]
        img = sitk.JoinSeries(volumes)
    else:
        img = sitk.GetImageFromArray(arr.astype(dtype))
    img.SetSpacing(spacing[::-1])
    if direction is not None:
        img.SetDirection(direction)
    if origin is not None:
        img.SetOrigin(origin)
    sitk.WriteImage(img, fname, True)


def extract_ids(file_list, f_suffix=".nii.gz"):
    new_list = []
    for fname in file_list:
        b_fname = os.path.basename(fname.strip(f_suffix))
        new_list.append(b_fname)

    return new_list


def create_acdc_abs_file_list(list_patids, root_dir):
    file_list = []
    for patid in list_patids:
        pat_dir = root_dir + '/patient{:03d}'.format(patid)
        file_list.append(pat_dir + '/patient{:03d}_4d.nii.gz'.format(patid))
    return file_list


def get_images_in_dir(src_path, dataset_name="ACDC", file_suffix=".nii.gz", rescale_int=False, do_downsample=False,
                      downsample_steps=None, int_perc=tuple((0, 100)), patid_list=None, transform=None,
                      resample_inplane=False, new_spacing=None):

    if dataset_name == "ACDC" and patid_list is not None and isinstance(patid_list[0], str):
        patid_list = [int(p_id.replace("patient", "")) for p_id in patid_list]
    # patid_list should be list of integers or "patient001" strings
    src_path = os.path.expanduser(src_path)
    filepath_generator = Path(src_path).rglob('*' + file_suffix)
    filepath_list = [file_obj for file_obj in filepath_generator]
    if len(filepath_list) == 0:
        raise ValueError("ERROR - get_images - no files found in {}".format(src_path + os.sep + '*' + file_suffix))
    filepath_list.sort()
    image_dict = {}
    print("INFO - get_images_in_dir - dataset name {} (#{}). Resample-inplane {}".format(dataset_name, len(filepath_list),
                                                                                         resample_inplane))

    for file_obj in filepath_list:
        patient_id = file_obj.name.replace(file_suffix, "")
        if dataset_name == "OASIS":
            patient_id = "_".join(file_obj.name.split("_")[:3])
            # OAS1_0452_MR1
            patid_search = int(patient_id.replace("OAS1_", "").replace("_MR1", "").replace("_MR2", ""))

        elif dataset_name in ['dHCP', 'MNIST3D', 'MNISTRoto']:
            patient_id = int(patient_id)
            patid_search = patient_id
        elif dataset_name == 'ADNI':
            # patient_id is string with absolute file name
            # if we remove src_path, the first subdir actually indicates pat ID
            subdir_list = str(file_obj.resolve()).replace(src_path, "").split(os.sep)
            patient_id = subdir_list[0] if subdir_list[0] != "" else subdir_list[1]
            seqno = subdir_list[-2] if subdir_list[-2] != "" else subdir_list[-3]
            if patient_id == seqno:
                # ach ja, difficult to make things generic
                # when loading original ADNI files 1mm and 6mm we need to create combined pat+seqno ids.
                # e.g. when loading upsample results directory naming already incorporates this concatenation
                pass
            else:
                patient_id = patient_id + "_" + seqno
            patid_search = patient_id
            # print(patient_id)
        elif dataset_name == "ACDC":
            if isinstance(patient_id, str):
                if "patient" in patient_id:
                    patient_id = int(patient_id.replace("patient", ""))
            patid_search = patient_id
        if patid_list is not None:
            if patid_search not in patid_list:
                continue
        # print("INFO - get_images_in_dir - Loading {}".format(str(file_obj.resolve())))
        img = sitk.ReadImage(str(file_obj.resolve()))
        orig_spacing = np.array(img.GetSpacing()[::-1]).astype(np.float64)
        np_image = sitk.GetArrayFromImage(img).astype(np.float32)
        if do_downsample:
            np_image = np_image[::int(downsample_steps)]
        if resample_inplane:
            spacing = np.array([orig_spacing[0], new_spacing[1], new_spacing[2]]).astype(np.float64)
            if np_image.ndim == 3:
                np_image = apply_2d_zoom_3d(np_image, orig_spacing, new_spacing=new_spacing, do_blur=True)
            else:  # assuming 4D then
                np_image = apply_2d_zoom_4d(np_image, orig_spacing, new_spacing=new_spacing, do_blur=True)

        else:
            spacing = orig_spacing
        if rescale_int:
            np_image = rescale_intensities(np_image, percs=int_perc)
        if transform is not None:
            np_image = transform({'image': np_image})['image']
        image_dict[patient_id] = {'image': np_image,
                                  'spacing': spacing, 'orig_spacing': orig_spacing,
                                  'origin': img.GetOrigin(), 'direction': img.GetDirection(),
                                  'patient_id': patient_id, "num_slices": np_image.shape[0] if np_image.ndim == 3 else np_image.shape[1]}
    return image_dict


def get_arvc_datasets(split=(0.70, 0.10, 0.20), rs=None, force=False) -> dict:
    """
    Creates three list with absolute file names of short-axis MRI images for ARVC dataset
    training, validation and test based on the specified split percentages.

    IMPORTANT: we first check whether we already created a certain split (split file name exists)
                if true, we load the existing file else we create a new one in data root dir e.g. ~/data/ARVC/

    :param split:
    :param rs:
    :param force:
    :return:
    """

    def create_absolute_file_names(rel_file_list, src_path) -> list:
        return [tuple((os.path.join(src_path, val[0]), val[1])) for val in rel_file_list]

    def get_dataset_files(all_files, file_ids) -> list:
        return [all_files[fid] for fid in file_ids]

    def combine_ids(original_id, pseudo_id) -> list:
        return [c_ids for c_ids in zip(original_id, pseudo_id)]

    def numpy_array_to_native_python(np_arr) -> list:
        return [val.item() for val in np_arr]

    dta_settings = get_config('ARVC')

    if os.path.isfile(dta_settings.split_file) and not force:
        print("INFO - get_arvc_datasets - Get split file from {}".format(dta_settings.split_file))
        # load existing splits
        with open(dta_settings.split_file, 'r') as fp:
            split_config = yaml.load(fp, Loader=yaml.FullLoader)
            training_ids = split_config['training']
            validation_ids = split_config['validation']
            test_ids = split_config['test']
        # print("INFO - Load existing split file {}".format(dta_settings.split_file))
    else:
        # create new split
        assert sum(split) == 1.
        # get a list with the short-axis image files that we have in total (e.g. in ~/data/ARVC/images/*.nii.gz)
        search_suffix = "*" + dta_settings.img_file_ext
        search_mask_img = os.path.expanduser(os.path.join(dta_settings.short_axis_dir, search_suffix))
        # we make a list of relative file names (root data dir is omitted)
        files_to_load = [os.path.basename(abs_fname) for abs_fname in get_image_file_list(search_mask_img)]
        num_of_patients = len(files_to_load)
        # permute the list of all files, we will separate the permuted list into train, validation and test sets
        if rs is None:
            rs = np.random.RandomState(78346)
        ids = rs.permutation(num_of_patients)
        # create three lists of files
        patids_train = numpy_array_to_native_python(ids[:int(split[0] * num_of_patients)])
        training_ids = get_dataset_files(files_to_load, patids_train)
        c_size = int(len(training_ids))
        patids_validation = numpy_array_to_native_python(ids[c_size:c_size + int(split[1] * num_of_patients)])
        validation_ids = get_dataset_files(files_to_load, patids_validation)
        c_size += len(validation_ids)
        patids_test = numpy_array_to_native_python(ids[c_size:])
        test_ids = get_dataset_files(files_to_load, patids_test)

        # write split configuration
        split_config = {'training': combine_ids(training_ids, patids_train),
                        'validation': combine_ids(validation_ids, patids_validation),
                        'test': combine_ids(test_ids, patids_test)}
        print("INFO - Write split file {}".format(dta_settings.split_file))
        with open(dta_settings.split_file, 'w') as fp:
            yaml.dump(split_config, fp)

    return {'training': create_absolute_file_names(training_ids, dta_settings.short_axis_dir),
            'validation': create_absolute_file_names(validation_ids, dta_settings.short_axis_dir),
            'test': create_absolute_file_names(test_ids, dta_settings.short_axis_dir)}


def rescale_intensities(im, dtype=np.float32, percs=tuple((0, 100))):
    min_val, max_val = np.percentile(im, percs)
    if np.isnan(min_val):
        print("WARNING - rescale_intensities - invalid min_val ", min_val)
        min_val = 0
    if np.isnan(max_val):
        max_val = 1

    im = ((im.astype(dtype) - min_val) / (max_val - min_val)).clip(0, 1)
    return im