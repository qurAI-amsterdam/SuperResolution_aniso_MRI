import os
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets.common import MyRandomSampler
from datasets.common import get_random_adjacent_slice
from torchvision import transforms
from datasets.shared_transforms import AdjustToPatchSize, RandomIntensity
from datasets.shared_transforms import RandomRotation, GenericToTensor, RandomCrop, CenterCrop
from tqdm import tqdm_notebook, tqdm
import SimpleITK as sitk
from pathlib import Path
from datasets.common import rescale_intensities
from kwatsch.common import isnotebook
from scipy.ndimage.filters import gaussian_filter1d
from datasets.data_config import get_config


def get_file_suffix_blurred(dataset_name, file_suffix, downsample_steps):
    if dataset_name == "OASIS":
        t_suffix = file_suffix.replace(".nii.gz", "")
        file_suffix_new = t_suffix + "_{}mm.nii.gz".format(downsample_steps)
    elif dataset_name == "dHCP":
        # slices have 0.5mm thickness. Hence, downsampling steps need to be divided by 2
        t_suffix = file_suffix.replace(".nii.gz", "")
        file_suffix_new = t_suffix + "_{:.1f}mm.nii.gz".format(downsample_steps / 2)
    elif dataset_name == "MNIST3D":
        return file_suffix
    elif dataset_name == 'ADNI':
        file_suffix_new = "_{}mm.nii".format(downsample_steps)
        return file_suffix_new
    else:
        raise NotImplementedError("Error - get_file_suffix_blurred - {} dataset not supported!".format(dataset_name))
    return file_suffix_new


def simulate_thick_slices(img3d, slice_thickness):
    # relate Gaussian standard deviation to FWHM of PSF: FWHM = 2.355 * sigma
    sigma = slice_thickness / 2.355
    img_lr = np.zeros_like(img3d)
    for y in range(img3d.shape[1]):
        for x in range(img3d.shape[2]):
            img_lr[:, y, x] = gaussian_filter1d(img3d[:, y, x], sigma)
    return img_lr


def get_transforms_brain(dataset, rs=np.random.RandomState(1234), patch_size=None, aug_patch_size=None):
    assert dataset in ['OASIS', 'dHCP', 'ADNI', 'MNIST3D', 'MNISTRoto']
    if dataset in ['dHCP']:
        # for dHCP all images have matrix 256x256. Hence, we don't need
        transform_tr = transforms.Compose([RandomRotation(rs=rs), RandomIntensity(rs=rs),
                                           GenericToTensor()])
        if patch_size < 256:
            print("!!! WARNING !!! --- USING RANDOM CROP ({}) FOR {} TRAINING !!! ".format(patch_size, dataset))
            transform_tr = transforms.Compose([
                    RandomCrop(patch_size, rs=rs), RandomRotation(rs=rs),
                                                  RandomIntensity(rs=rs), GenericToTensor()])
        transform_te = transforms.Compose([GenericToTensor()])
    elif dataset in ['ADNI']:
        # for ADNI all images have different matrix size, but we adjust to 256x256
        if patch_size < 256:
            print("!!! WARNING !!! --- USING RANDOM CROP ({}) FOR OASIS TRAINING !!! ".format(patch_size))
            transform_tr = transforms.Compose([AdjustToPatchSize(tuple((256, 256))),
                                               RandomCrop(patch_size, rs=rs),
                                               RandomRotation(rs=rs), RandomIntensity(rs=rs),
                                               GenericToTensor()])
        else:
            transform_tr = transforms.Compose([AdjustToPatchSize(tuple((patch_size, patch_size))),
                                               RandomRotation(rs=rs), RandomIntensity(rs=rs),
                                               GenericToTensor()])
        transform_te = transforms.Compose([AdjustToPatchSize(tuple((patch_size, patch_size))), GenericToTensor()])
    elif dataset == 'OASIS':
        # AdjustToPatchSize
        assert patch_size is not None
        if patch_size < 220:
            print("!!! WARNING !!! --- USING RANDOM CROP ({}) FOR OASIS TRAINING !!! ".format(patch_size))
            transform_tr = transforms.Compose([AdjustToPatchSize(tuple((220, 220))),
                                               RandomCrop(patch_size, rs=rs),
                                               RandomRotation(rs=rs), RandomIntensity(rs=rs),
                                               GenericToTensor()])
        else:
            transform_tr = transforms.Compose([AdjustToPatchSize(tuple((patch_size, patch_size))),
                                               RandomRotation(rs=rs), RandomIntensity(rs=rs),
                                               GenericToTensor()])
        transform_te = transforms.Compose([AdjustToPatchSize(tuple((patch_size, patch_size))), GenericToTensor()])
    elif dataset in ["MNIST3D", 'MNISTRoto']:
        if aug_patch_size is not None:
            print("INFO - {} - Transforms - AdjustToPatchSize".format(dataset))
            # transform_tr = transforms.Compose([AdjustToPatchSize(tuple((aug_patch_size, aug_patch_size))),
            #                                    RandomIntensity(rs=rs), GenericToTensor()])
            transform_tr = transforms.Compose([AdjustToPatchSize(tuple((aug_patch_size, aug_patch_size))),
                                               GenericToTensor()])
            transform_te = transforms.Compose([AdjustToPatchSize(tuple((aug_patch_size, aug_patch_size))),
                                               GenericToTensor()])
        else:
            print("INFO - {} - Transforms ".format(dataset))
            # transform_tr = transforms.Compose([RandomIntensity(rs=rs), GenericToTensor()])
            transform_tr = transforms.Compose([GenericToTensor()])
            transform_te = transforms.Compose([GenericToTensor()])
    return transform_tr, transform_te


def get_data_loaders_brain_dataset(dataset, args, use_cuda=True, shuffle=True):
    loader = None
    kwargs = {'num_workers': 2} if use_cuda else {}
    if dataset.dataset.lower() == "training":
        # compute total number of iters for main loop
        # num_of_iters = args.batch_size * len(dataset)
        sampler = MyRandomSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], sampler=sampler, drop_last=True, **kwargs)
    if dataset.dataset.lower() == "test" or dataset.dataset.lower() == "validation":
        loader = torch.utils.data.DataLoader(dataset, batch_size=args['test_batch_size'], shuffle=shuffle, drop_last=True, **kwargs)

    return loader


def determine_interpol_coefficients(sliceid_from, sliceid_to, sliceid_between):
    gap = sliceid_to - sliceid_from
    return 1 - ((sliceid_between - sliceid_from) * 1/gap), 1 - ((sliceid_to - sliceid_between) * 1/gap)


def get_adni_patient_id(abs_filename: str) -> str:
    # patient_id is string with absolute file name
    # if we remove src_path, the first subdir actually indicates pat ID
    subdir_list = abs_filename.split(os.sep)
    patient_id = subdir_list[0] if subdir_list[0] != "" else subdir_list[1]
    seqno = subdir_list[-2] if subdir_list[-2] != "" else subdir_list[-3]
    return patient_id + "_" + seqno


"""
    CURRENTLY USED FOR OASIS dataset 
"""


def process_img(np_image, transform, do_downsample, downsample_steps, rescale_int, int_perc=tuple((0, 100))):

    if transform is not None:
        np_image = transform({'image': np_image})['image']
    if do_downsample:
        np_image = np_image[::int(downsample_steps)]
    if rescale_int:
        np_image = rescale_intensities(np_image, percs=int_perc)
    return np_image


def get_images(patid_list, dataset, rescale_int=False, int_perc=tuple((0, 100)), limited_load=False,
               do_downsample=False, downsample_steps=3, transform=None, include_hr_images=False,
               verbose=False, hr_only=False, src_path=None):
    """
        In case "do_donwsample=False" we only load high resolution images
        When downsampling, we can choose to load both, blurred dataset and HR dataset
    """
    assert dataset in ['OASIS', 'dHCP', 'MNIST3D', 'MNISTRoto', 'ADNI']
    data_config = get_config(dataset)
    file_suffix = data_config.img_file_ext
    if src_path is None:
        src_path = os.path.expanduser(data_config.image_dir)
    else:
        src_path = os.path.expanduser(src_path)
    file_suffix_blurred = get_file_suffix_blurred(dataset, file_suffix, downsample_steps)
    if verbose:
        print("INFO - get_images - using file suffix for search {}".format(file_suffix_blurred))
    if hr_only:
        # currently only used when creating new dataset with lower resolution
        print("WARNING - common_brains.get_images - retrieving HR images for {} from {} mask {}".format(dataset,
                                                                                                        src_path,
                                                                                                        '*' + file_suffix))
        filepath_generator = Path(src_path).rglob('*' + file_suffix)
    else:
        filepath_generator = Path(src_path).rglob('*' + file_suffix_blurred)
    file_suffix_hr = file_suffix if (include_hr_images or hr_only) else None
    if dataset == "OASIS":
        filepath_list = [file_obj for file_obj in filepath_generator if int(file_obj.name.split("_")[1]) in patid_list]
    elif dataset.lower() == 'dhcp':
        filepath_list = [file_obj for file_obj in filepath_generator if int(file_obj.name.split("_")[0]) in patid_list]
    elif dataset.lower() in ['mnist3d', 'mnistroto']:
        filepath_list = [file_obj for file_obj in filepath_generator if int(file_obj.name.replace(file_suffix, "")) in patid_list]
    elif dataset.lower() == 'adni':
        filepath_list = [file_obj for file_obj in filepath_generator if get_adni_patient_id(str(file_obj.resolve()).replace(src_path, "")) in patid_list]
        # filepath_list = [file_obj for file_obj in filepath_generator]
    else:
        raise NotImplementedError()
    if len(filepath_list) == 0:
        raise ValueError("ERROR - get_images - no files found in {}".format(src_path))
    filepath_list.sort()
    if limited_load:
        filepath_list = filepath_list[:data_config.limited_load_max]
    image_dict = {}
    tqdm_func = tqdm_notebook if isnotebook() else tqdm
    shapes = []
    for file_obj in tqdm_func(filepath_list, desc="Loading {} volumes from {}".format(len(filepath_list), src_path)):
        if dataset.lower() == "dhcp":
            patient_id = int(file_obj.name.replace(file_suffix, "").split("_")[0])
        elif dataset.lower() == 'oasis':
            patient_id = "_".join(file_obj.name.split("_")[:3])
            # print(patient_id)
        elif dataset == 'ADNI':
            patient_id = get_adni_patient_id(str(file_obj.resolve()).replace(src_path, ""))
        elif dataset.lower() == 'mnist3d':
            patient_id = int(file_obj.name.replace(file_suffix, ""))
        else:
            raise ValueError("Error - dataset {} currently not supported".format(dataset))
        # if verbose:
            # print("INFO - get_images - {} - load {}".format(dataset, str(file_obj.resolve())))
        img = sitk.ReadImage(str(file_obj.resolve()))
        np_image = sitk.GetArrayFromImage(img).astype(np.float32)
        np_image = process_img(np_image, transform, do_downsample, downsample_steps, rescale_int, int_perc)
        np_hr_image = None
        if include_hr_images:
            img_hr = sitk.ReadImage(str(file_obj.resolve()).replace(file_suffix_blurred, file_suffix_hr))
            np_hr_image = sitk.GetArrayFromImage(img_hr).astype(np.float32)
            np_hr_image = process_img(np_hr_image, transform, do_downsample, downsample_steps, rescale_int, int_perc)
        shapes.append(np_image.shape)
        image_dict[patient_id] = {'image': np_image, 'image_hr': np_hr_image,
                                  'spacing': np.array(img.GetSpacing()[::-1]).astype(np.float64),
                                  'origin': img.GetOrigin(), 'direction': img.GetDirection(),
                                  'patient_id': patient_id, "num_slices": np_image.shape[0],
                                  'img_path': str(file_obj.resolve())}
    shapes = np.array(shapes)
    print(np.mean(shapes, axis=0))
    return image_dict


class BrainDataset(Dataset):

    def _get_indices(self):
        allidcs = np.empty((0, 3), dtype=int)
        for patnum, image_dict in self.images.items():
            num_slices = image_dict['num_slices']
            img_nbr = np.repeat(patnum, num_slices)
            slice_range = np.arange(0, num_slices)
            slice_max_num = np.repeat(num_slices, num_slices)
            allidcs = np.vstack((allidcs, np.vstack((img_nbr, slice_range, slice_max_num)).T))

        self._idcs = allidcs  # .astype(int)

    def __len__(self):
        return len(self._idcs)

    def __getitem__(self, idx):
        patnum, slice_id_1, num_slices = self._idcs[idx]
        slice_id_1, num_slices = int(slice_id_1), int(num_slices)
        slice_step = self._get_slice_step()
        slice_id_2 = get_random_adjacent_slice(slice_id_1, num_slices, rs=self.rs, step=slice_step)
        inbetween_slice_id, is_inbetween = self._get_inbetween_sliceid(slice_id_1, slice_id_2)
        if self.rs.choice([0, 1]) == 0:
            slice_idx_from, slice_idx_to = slice_id_1, slice_id_2
        else:
            slice_idx_from, slice_idx_to = slice_id_2, slice_id_1
        alpha_from, alpha_to = determine_interpol_coefficients(slice_idx_from, slice_idx_to, inbetween_slice_id)
        # print("braindHCP - getitem - {}-{}-{}  {:.2f} {:.2f}".format(slice_idx_from, inbetween_slice_id,
        #                                                           slice_idx_to, alpha_from, alpha_to))
        img = np.vstack((self.images[patnum]['image'][slice_idx_from][None],
                         self.images[patnum]['image'][slice_idx_to][None],
                         self.images[patnum]['image'][inbetween_slice_id][None]))
        sample = {'image': img, 'patient_id': patnum, 'num_slices_vol': num_slices,
                  'slice_idx_from': slice_idx_from, 'slice_idx_to': slice_idx_to,
                  'alpha_from': np.array([alpha_from]).astype(np.float32),
                  'alpha_to': np.array([alpha_to]).astype(np.float32),
                  'inbetween_slice_id': inbetween_slice_id,
                  'is_inbetween': np.float32(is_inbetween)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def set_transform(self, transform):
        self.transform = transform

    def _get_slice_step(self):
        if self.slice_selection == "adjacent":
            return 1
        elif self.slice_selection == "adjacent_plus":
            return self.downsample_steps
        elif self.slice_selection == "mix":
            return self.rs.choice([1, self.downsample_steps])

    def _get_inbetween_sliceid(self, sliceid1, sliceid2):
        in_between_sliceid = self.rs.choice(np.arange(min(sliceid1, sliceid2) + 1, max(sliceid1, sliceid2)))
        return in_between_sliceid, 1


def prepare_batch_pairs(batch_dict, expand_type="repeat"):
    """

    :param batch_dict: batch_dict['image'] has shape [batch, 2, h, w]
            channel 0 = patient slice i
            channel 1 = patient slice i+1 (or i -1)
    :param expand_type: 'repeat' or 'reshape' or 'split'
    :return: split
    """
    batch_images = batch_dict['image']
    assert batch_images.size(0) % 2 == 0
    if expand_type == "repeat" or expand_type == "split":
        """
            Numpy repeat functionality for batch items. We just concatenate everything from dim1 (two adjacent mri scans)
            so e.g. patient016, frame 2, slice 2 and 3 are separated by #batches
        """
        a = torch.unsqueeze(batch_images[:, 0], dim=1)
        b = torch.unsqueeze(batch_images[:, 1], dim=1)
        if batch_images.shape[1] == 3:
            batch_dict['slice_between'] = torch.unsqueeze(batch_images[:, 2], dim=1)
        if expand_type == "split":
            batch_dict['image_from'], batch_dict['image_to'] = a, b

        else:
            batch_dict['image'] = torch.cat([a, b], dim=0)
            # batch_dict['slice_id_from'] = torch.cat([batch_dict['slice_id1'], batch_dict['slice_id1']])
            # batch_dict['slice_id_to'] = torch.cat([batch_dict['slice_id2'], batch_dict['slice_id2']])
            # batch_dict['inbetween_slice_id'] = torch.cat([batch_dict['inbetween_slice_id'],
            #                                               batch_dict['inbetween_slice_id']])
            # batch_dict['patient_id'] = torch.cat([batch_dict['patient_id'], batch_dict['patient_id']])
            # batch_dict['is_inbetween'] = torch.cat([batch_dict['is_inbetween'], batch_dict['is_inbetween']])
            # batch_dict['num_slices_vol'] = torch.cat([batch_dict['num_slices_vol'], batch_dict['num_slices_vol']])

    else:
        raise ValueError("Error - prepare_batch_pairs - valid values for expand_type parameter are repeat, "
                         "reshape, split.")
    return batch_dict