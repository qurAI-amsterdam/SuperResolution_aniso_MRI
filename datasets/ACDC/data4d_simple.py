import os
import torch
import numpy as np
import SimpleITK as sitk
from _collections import defaultdict
from tqdm import tqdm, tqdm_notebook
from torchvision import transforms
from torch.utils.data.sampler import RandomSampler, Sampler
from collections import OrderedDict
from kwatsch.common import isnotebook
from datasets.data_config import get_config
from datasets.ACDC.data import ACDCImage, get_acdc_patient_ids, get_patids_acdc_sr
from torch.utils.data import Dataset
from datasets.ACDC.data4d import create_file_list
from datasets.ACDC.acdc_transforms import ToTensor, CenterCrop, RandomAnyRotation
from datasets.common import apply_2d_zoom_3d, get_random_adjacent_slice
from datasets.common import MyRandomSampler


def get_data_loaders_acdc(dataset, args, rs=None, use_cuda=True, shuffle=True):
    loader = None
    kwargs = {'num_workers': 2} if use_cuda else {}
    if dataset.dataset == "training":
        # compute total number of iters for main loop
        # num_of_iters = args.batch_size * len(dataset)
        sampler = MyRandomSampler(dataset, rs)  # RandomSampler(dataset, replacement=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], sampler=sampler, drop_last=True, **kwargs)
    if dataset.dataset == "validation":
        loader = torch.utils.data.DataLoader(dataset, batch_size=args['test_batch_size'], shuffle=shuffle, drop_last=True, **kwargs)

    return loader


def get_new_dataset_acdc(args, dta_settings, rs, acdc_set="both", new_spacing=None, transform_tr=None,
                         transform_te=None, test_limited_load=True, get_masks=False):
    training_set = None
    val_set = None
    if transform_tr is None:
        transform_tr = transforms.Compose([CenterCrop(args['width']), RandomAnyRotation(rs=rs, max_degree=359),
                                                                         MyToTensor()])
    if transform_te is None:
        transform_te = transforms.Compose([CenterCrop(args['width']), MyToTensor()])
    if acdc_set in ['both', 'train']:
        training_set = ACDCDataset4DPairs('training',
                                           root_dir=dta_settings.short_axis_dir,
                                           resample=True,
                                           transform=transform_tr,
                                           limited_load=args['limited_load'],
                                          slice_selection=args['slice_selection'],
                                          new_spacing=new_spacing,
                                          rs=rs,
                                          get_masks=get_masks)
    if acdc_set in ['both', 'test']:
        val_set = ACDCDataset4DPairs('validation',
                                      root_dir=dta_settings.short_axis_dir,
                                      resample=True,
                                      transform=transform_te,
                                      limited_load=test_limited_load,
                                     slice_selection=args['slice_selection'],
                                     new_spacing=new_spacing,
                                     rs=rs,
                                     get_masks=get_masks)
    return training_set, val_set


def apply_transform(image_dict, transform):
    # assuming numpy arrays of 4D [t, z, y, x]. Also image_dict has 'image' and may be 'mask' key
    for p_id, data_dict in image_dict.items():
        new_array4d = None
        new_mask4d, new_labels4d = None, None
        do_mask = True if 'mask' in data_dict.keys() else False
        do_labels = True if 'labels' in data_dict.keys() else False
        for f_id in np.arange(data_dict['image'].shape[0]):
            t_arr3d = transform({'image': data_dict['image'][f_id]})['image']
            new_array4d = np.vstack((new_array4d, t_arr3d[None])) if new_array4d is not None else t_arr3d[None]
            if do_mask:
                t_m_arr3d = transform({'image': data_dict['mask'][f_id]})['image']
                new_mask4d = np.vstack((new_mask4d, t_m_arr3d[None])) if new_mask4d is not None else t_m_arr3d[None]
            if do_labels:
                t_m_arr3d = transform({'image': data_dict['labels'][f_id]})['image']
                new_labels4d = np.vstack((new_labels4d, t_m_arr3d[None])) if new_labels4d is not None else t_m_arr3d[None]
        image_dict[p_id]['image'] = new_array4d
        if do_mask:
            image_dict[p_id]['mask'] = new_mask4d
        if do_labels:
            image_dict[p_id]['labels'] = new_labels4d
    return image_dict


def get_4d_image_array(root_dir, dataset=None, rescale=True, resample=False, limited_load=False, new_spacing=None,
                       rs=np.random.RandomState(1234), pat_nums=None,
                       get_masks=False, transform=None):
    # pat_nums: list of integers
    root_dir = os.path.expanduser(root_dir)
    print("WARNING - get_4d_image_array - Loading ACDC data from {}".format(root_dir))
    assert (dataset is None and pat_nums is not None) or (dataset is not None and pat_nums is None)
    if pat_nums is None:
        pat_nums = get_patids_acdc_sr(dataset, rs=rs, limited_load=limited_load, max_limit_load=2)
        load_info = 'Load {} set'.format(dataset)
    else:
        load_info = "Loading {} patients".format(len(pat_nums))
    abs_file_list = create_file_list(pat_nums, root_dir,
                                     include_arvc=False, limited_load=limited_load)
    abs_file_list.sort()
    dataset_dict = OrderedDict()

    if isnotebook():
        myiterator = tqdm_notebook(enumerate(abs_file_list), desc=load_info, total=len(pat_nums))
    else:
        myiterator = tqdm(enumerate(abs_file_list), desc=load_info, total=len(pat_nums))
    for idx, abs_filename in myiterator:
        patnum = pat_nums[idx]
        img = ACDCImage(patnum, root_dir=None, resample=resample, scale_intensities=rescale,
                        abs_filename=abs_filename, new_spacing=new_spacing)
        image4d_dict = img.preprocessed4d()
        if get_masks:
            mask_dict = get_4d_acdc_masks("~/data/ACDC4d_masks/", None, None, resample=resample, rs=rs,
                                          new_spacing=tuple((1, 1.4, 1.4)), pat_nums=[patnum])
            image4d_dict['mask'] = mask_dict[patnum]['image']

        dataset_dict[patnum] = image4d_dict
    if transform is not None:
        dataset_dict = apply_transform(dataset_dict, transform)
    return dataset_dict


def get_normalized_frame_slice_info(frame_id, slice_id, num_frames, num_slices):
    return (frame_id + 1) / num_frames,  (slice_id + 1) / num_slices


class ACDCDataset4DPairs(Dataset):

    """
            Parameter adjacency:

            Random in this context means that "paired_image" only contains one image slice and not TWO which is the case in
            "paired" and "combined" mode. The Pytorch dataloader will create batches with size 1 in dimension 1, whereas in the
            other modes it will return size 2 in dimension 1. We use the function "prepare_batch_pairs" to reshuffle the batches
            so that slices for interpolation will be generated by splitting the batch into two halves. Hence, in those cases the
            batch size doubles to batch_size * 2. In random mode the batch size stays the same and for interpolation the batch is
            just splitted.
    """

    def __init__(self, dataset,  # ['training', 'validation', 'full', 'test']  test: only applicable for SR
                 images4d=None,
                 root_dir='~/data/ACDC/all_cardiac_phases',
                 transform=None, limited_load=False,
                 rescale=True,
                 resample=False,
                 slice_selection="mix",
                 new_spacing=None, rs=np.random.RandomState(1234), get_masks=False):
        assert slice_selection in ['adjacent', 'adjacent_plus', 'mix']
        self._root_dir = root_dir
        self.transform = transform
        self._resample = resample
        self._get_masks = get_masks
        self.slice_selection = slice_selection
        self.dataset = dataset
        self.z_spacings = defaultdict(list)
        self.rs = rs
        if images4d is None:
            self.images4d = get_4d_image_array(root_dir, dataset, rescale=rescale, resample=resample, limited_load=limited_load,
                                               new_spacing=new_spacing, rs=rs,
                                               get_masks=get_masks)
        else:
            self.images4d = images4d
        self._get_indices()
        # print("INFO - ACDCDataset4DPairs - unique z-spacings ", self.z_spacings.keys())

    def _get_indices(self):
        allidcs = np.empty((0, 4), dtype=int)
        for patnum, image_dict in self.images4d.items():
            num_frames = image_dict['num_frames']
            num_slices = image_dict['num_slices']
            self.z_spacings[image_dict['spacing'][0]].append("patient_id{:03d}".format(patnum))
            # JS changed 01-07: slice_to and slice_from have length num_slices - 2
            # img_nbr = np.repeat(patnum, num_frames * num_slices)
            img_nbr = np.repeat(patnum, num_frames * num_slices)
            frame_range = np.arange(0, num_frames)
            frame_range = np.tile(frame_range, num_slices)
            slice_range = np.arange(0, num_slices)
            slice_range = np.repeat(slice_range, num_frames)
            slice_max_num = np.repeat(num_slices, num_frames * num_slices)
            allidcs = np.vstack((allidcs, np.vstack((img_nbr, frame_range, slice_range, slice_max_num)).T))

        self._idcs = allidcs.astype(int)

    def __len__(self):
        return len(self._idcs)

    def __getitem__(self, idx):
        patnum, frame_idx_from, slice_id_1, num_slices = self._idcs[idx]
        f, s, _, _ = self.images4d[patnum]['image'].shape
        # For pat 15, 34, 45 due to voxel intensity errors we skip frames 20-29. But we need original num_frames
        # as feature for alpha_probe network
        orig_num_frames = self.images4d[patnum]['orig_num_frames']
        slice_step = self._get_slice_step()
        slice_id_2 = get_random_adjacent_slice(slice_id_1, num_slices, rs=self.rs, step=slice_step)
        inbetween_slice_id, is_inbetween = self._get_inbetween_sliceid(slice_id_1, slice_id_2)
        if self.rs.choice([0, 1]) == 0:
            slice_idx_from, slice_idx_to = slice_id_1, slice_id_2
        else:
            slice_idx_from, slice_idx_to = slice_id_2, slice_id_1

        if self._get_masks:
            loss_mask = self.images4d[patnum]['mask'][frame_idx_from][None, inbetween_slice_id]
        else:
            loss_mask = np.array([1]).astype(np.float32)
        # print("slice IDs ", s, slice_idx_from, inbetween_slice_id, slice_idx_to)
        try:
            paired_image = np.vstack((np.expand_dims(self.images4d[patnum]['image'][frame_idx_from][slice_idx_from], 0),
                                      np.expand_dims(self.images4d[patnum]['image'][frame_idx_from][slice_idx_to], 0),
                                      np.expand_dims(self.images4d[patnum]['image'][frame_idx_from][inbetween_slice_id], 0)))
        except:
            raise ValueError(self.images4d[patnum]['patient_id'], idx, frame_idx_from, f,
                             self.images4d[patnum]['image'].shape, "mix slice id {}".format(inbetween_slice_id))

        num_slices = self.images4d[patnum]['num_slices']
        # if self.dataset == 'training':
        #    print("ACDCDataset4DPairs ", self.dataset, idx, patnum, frame_idx_from, slice_idx_from, slice_idx_to)
        sample = {'image': paired_image,
                  'num_slices_vol': np.array([num_slices]).astype(np.float32),
                  'num_frames_vol': np.array([orig_num_frames]).astype(np.float32),
                  'slice_id_from': np.array([slice_idx_from]).astype(np.float32),
                  'slice_id_to': np.array([slice_idx_to]).astype(np.float32),
                  'spacing': self.images4d[patnum]['spacing'],
                  'frame_id_from': np.array([frame_idx_from]).astype(np.float32),
                  'frame_id_to': np.array([frame_idx_from]).astype(np.float32),
                  'patient_id': np.array([patnum]).astype(np.int),
                  'original_spacing': self.images4d[patnum]['original_spacing'],
                  'is_inbetween': np.float32(is_inbetween),
                  'loss_mask': loss_mask,
                  'alpha_from': np.array([0.5]).astype(np.float32),
                  'alpha_to': np.array([0.5]).astype(np.float32),
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def set_transform(self, transform):
        self.transform = transform

    def _get_slice_step(self):
        if self.slice_selection == "adjacent":
            return 1
        elif self.slice_selection == "adjacent_plus":
            return 2
        elif self.slice_selection == "mix":
            return self.rs.choice([1, 2])

    @staticmethod
    def _get_inbetween_sliceid(sliceid1, sliceid2):
        if (sliceid1 + sliceid2) % 2 == 0:
            # gap between slice ids is one.
            in_between_sliceid = (sliceid1 + sliceid2) // 2
            is_inbetween = 1
        else:
            in_between_sliceid = sliceid1
            is_inbetween = 0
        return in_between_sliceid, is_inbetween


class ACDCDataset4DPairsPatient(ACDCDataset4DPairs):

    def __init__(self, dataset,  # ['training', 'validation', 'full', 'test']  test: only applicable for SR
                 images4d=None,
                 root_dir=os.path.expanduser('~/data/ACDC/all_cardiac_phases'),
                 transform=None, limited_load=False,
                 rescale=True,
                 resample=False,
                 slice_selection="mix",
                 new_spacing=None, rs=np.random.RandomState(1234), get_masks=False):
        super(ACDCDataset4DPairsPatient, self).__init__(dataset, images4d, root_dir, transform, limited_load, rescale, resample,
                                                        slice_selection, new_spacing, rs, get_masks)

    def __getitem__(self, idx):
        do_continue = True
        save_exit = 0
        while do_continue:
            patnum, frame_idx_from, slice_idx_from, num_slices = self._idcs[idx]
            save_exit += 1
            if slice_idx_from % 2 == 0 and slice_idx_from != num_slices - 1:
                do_continue = False
                slice_idx_to = slice_idx_from + 2
            if save_exit > 100:
                do_continue = False
        f, s, _, _ = self.images4d[patnum]['image'].shape
        orig_num_frames = self.images4d[patnum]['orig_num_frames']
        inbetween_slice_id, is_inbetween = slice_idx_from + 1, 1
        if self._get_masks:
            loss_mask = self.images4d[patnum]['mask'][frame_idx_from][None, inbetween_slice_id]
        else:
            loss_mask = np.array([1]).astype(np.float32)
        # print("slice IDs ", s, slice_idx_from, inbetween_slice_id, slice_idx_to)
        try:
            paired_image = np.vstack((np.expand_dims(self.images4d[patnum]['image'][frame_idx_from][slice_idx_from], 0),
                                      np.expand_dims(self.images4d[patnum]['image'][frame_idx_from][slice_idx_to], 0),
                                      np.expand_dims(self.images4d[patnum]['image'][frame_idx_from][inbetween_slice_id],
                                                     0)))
        except:
            raise ValueError(self.images4d[patnum]['patient_id'], idx, frame_idx_from, f,
                             self.images4d[patnum]['image'].shape, "mix slice id {}".format(inbetween_slice_id))

        num_slices = self.images4d[patnum]['num_slices']

        sample = {'image': paired_image,
                  'num_slices_vol': np.array([num_slices]).astype(np.float32),
                  'num_frames_vol': np.array([orig_num_frames]).astype(np.float32),
                  'slice_id_from': np.array([slice_idx_from]).astype(np.float32),
                  'slice_id_to': np.array([slice_idx_to]).astype(np.float32),
                  'spacing': self.images4d[patnum]['spacing'],
                  'frame_id_from': np.array([frame_idx_from]).astype(np.float32),
                  'frame_id_to': np.array([frame_idx_from]).astype(np.float32),
                  'patient_id': np.array([patnum]).astype(np.int),
                  'original_spacing': self.images4d[patnum]['original_spacing'],
                  'is_inbetween': np.float32(is_inbetween),
                  'loss_mask': loss_mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


def prepare_batch_pairs(batch_dict, expand_type="repeat"):
    """

    :param batch_dict: batch_dict['image'] has shape [batch, 2, h, w]
            channel 0 = patient slice i
            channel 1 = patient slice i+1 (or i -1)
    :param device: 'cuda' or 'cpu'
    :param expand_type: 'repeat' or 'reshape' or 'split'
    :param rs: random state object from numpy
    :return: split
    """
    batch_images = batch_dict['image']
    # assert batch_images.size(0) % 2 == 0
    if expand_type == "repeat" or expand_type == "split":
        """
            !!!!  batch_images    [batch, 2, h, w] or [batch, 3, h, w]  or [batch, 6, h, w]
            Numpy repeat functionality for batch items. We just concatenate everything from dim1 (two adjacent mri scans)
            so e.g. patient016, frame 2, slice 2 and 3 are separated by #batches
        """
        if batch_images.shape[1] == 2 or batch_images.shape[1] == 3:
            a = torch.unsqueeze(batch_images[:, 0], dim=1)
            b = torch.unsqueeze(batch_images[:, 1], dim=1)
        else:
            # WE ASSUME 6 slices in dim1, slice 1 - 3 plus their seg labels
            a = batch_images[:, 0:2]
            b = batch_images[:, 2:4]
            batch_dict['slice_between'] = batch_images[:, 4:]
        if batch_images.shape[1] == 3:
            batch_dict['slice_between'] = torch.unsqueeze(batch_images[:, 2], dim=1)

        if expand_type == "split":
            batch_dict['image_from'], batch_dict['image_to'] = a, b
        else:
            batch_dict['image'] = torch.cat([a, b], dim=0)
            # batch_dict['slice_pair'] = torch.cat([batch_dict['slice_pair'], batch_dict['slice_pair']])
            # slice_id contains tensor of [b, 2]
            batch_dict['num_slices_vol'] = torch.cat([batch_dict['num_slices_vol'], batch_dict['num_slices_vol']])
            batch_dict['num_frames_vol'] = torch.cat([batch_dict['num_frames_vol'], batch_dict['num_frames_vol']])
            batch_dict['spacing'] = torch.cat([batch_dict['spacing'], batch_dict['spacing']])
            batch_dict['original_spacing'] = torch.cat([batch_dict['original_spacing'], batch_dict['original_spacing']])
            batch_dict['frame_id_from'] = torch.cat([batch_dict['frame_id_from'], batch_dict['frame_id_from']])
            batch_dict['frame_id_to'] = torch.cat([batch_dict['frame_id_to'], batch_dict['frame_id_to']])
            batch_dict['slice_id_from'] = torch.cat([batch_dict['slice_id_from'], batch_dict['slice_id_from']])
            batch_dict['slice_id_to'] = torch.cat([batch_dict['slice_id_to'], batch_dict['slice_id_to']])
            batch_dict['patient_id'] = torch.cat([batch_dict['patient_id'], batch_dict['patient_id']])
            batch_dict['is_inbetween'] = torch.cat([batch_dict['is_inbetween'], batch_dict['is_inbetween']])
            batch_dict['loss_mask'] = torch.cat([batch_dict['loss_mask'], batch_dict['loss_mask']])

    elif expand_type == 'single':
        # we will only take the first slice of each batch item
        # Get first item in dim1 (image has shape [b, 2, x, y]
        batch_dict['image'] = batch_dict['image'][:, 0, None]
        if batch_images.shape[1] == 3:
            batch_dict['slice_between'] = torch.unsqueeze(batch_images[:, 2], dim=1)
        # Shuffle rows
        batch_dict['image'] = batch_dict['image'][torch.randperm(batch_dict['image'].size(0))]

    else:
        raise ValueError("Error - prepare_batch_pairs - valid values for expand_type parameter are "
                         "repeat, reshape, split.")
    return batch_dict


class MyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        spacing, original_spacing = sample['spacing'], sample['original_spacing']
        patient_id, num_slices_vol = sample['patient_id'], sample['num_slices_vol']
        frame_id_from, frame_id_to = sample['frame_id_from'], sample['frame_id_to']
        slice_id_from, slice_id_to = sample['slice_id_from'], sample['slice_id_to']
        num_frames_vol, loss_mask = sample['num_frames_vol'], sample["loss_mask"]
        is_inbetween = None
        if "is_inbetween" in sample.keys():
            is_inbetween = sample["is_inbetween"]
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32)

        try:
            image = torch.from_numpy(image)
            spacing = torch.from_numpy(spacing)
            original_spacing = torch.from_numpy(original_spacing)
            patient_id = torch.from_numpy(patient_id)
            frame_id_from = torch.from_numpy(frame_id_from)
            frame_id_to = torch.from_numpy(frame_id_to)
            num_slices_vol = torch.from_numpy(num_slices_vol)
            num_frames_vol = torch.from_numpy(num_frames_vol)
            slice_id_from = torch.from_numpy(slice_id_from)
            slice_id_to = torch.from_numpy(slice_id_to)
            loss_mask = torch.from_numpy(loss_mask)
            if is_inbetween is not None:
                is_inbetween = torch.from_numpy(np.array([is_inbetween]).astype(np.float32))

        except ValueError:
            image = torch.from_numpy(np.ascontiguousarray(image))
            spacing = torch.from_numpy(np.ascontiguousarray(spacing))
            original_spacing = torch.from_numpy(np.ascontiguousarray(original_spacing))
            patient_id = torch.from_numpy(np.ascontiguousarray(patient_id))
            frame_id_from = torch.from_numpy(np.ascontiguousarray(frame_id_from))
            frame_id_to = torch.from_numpy(np.ascontiguousarray(frame_id_to))
            num_slices_vol = torch.from_numpy(np.ascontiguousarray(num_slices_vol))
            num_frames_vol = torch.from_numpy(np.ascontiguousarray(num_frames_vol))
            slice_id_from = torch.from_numpy(np.ascontiguousarray(slice_id_from))
            slice_id_to = torch.from_numpy(np.ascontiguousarray(slice_id_to))
            loss_mask = torch.from_numpy(np.ascontiguousarray(loss_mask))
            if is_inbetween is not None:
                is_inbetween = torch.from_numpy(np.array([is_inbetween]).astype(np.float32))

        new_sample = {'image': image,
                      'num_slices_vol': num_slices_vol,
                      'num_frames_vol': num_frames_vol,
                      'slice_id_from': slice_id_from,
                      'slice_id_to': slice_id_to,
                      'spacing': spacing,
                      'original_spacing': original_spacing,
                      'patient_id': patient_id,
                      'frame_id_from': frame_id_from,
                      'frame_id_to': frame_id_to,
                      'loss_mask': loss_mask}
        if "is_inbetween" in sample.keys():
            new_sample["is_inbetween"] = is_inbetween
        del sample
        return new_sample


def get_4d_acdc_masks(root_dir, fold, dataset, resample=False, limited_load=False, new_spacing=None,
                       thick_slices_only=False, rs=np.random.RandomState(1234), pat_nums=None):
    """
        Basically same as get_4d_image_array. But instead of images we load binary segmentation masks
        for RV, LVM and LV. Dilated 5 times. We use these to mask out unnecessary structures when computing
        evaluation metrics (and losses?)
    """
    # pat_nums: list of integers
    root_dir = os.path.expanduser(root_dir)
    # print("WARNING - get_4d_acdc_masks - Loading ACDC dilated masks from {}".format(root_dir))
    if pat_nums is None:
        if thick_slices_only:
            # only train on patients with slice thickness above 5mm. We use 5mm studies to evaluate SR approach
            pat_nums = get_patids_acdc_sr(dataset, rs=rs, limited_load=limited_load, max_limit_load=2)
        else:
            pat_nums = get_acdc_patient_ids(fold, dataset, limited_load=limited_load, max_limit_load=2)
    pat_nums.sort()
    abs_file_list = [os.path.join(root_dir, "patient{:03d}_4d.nii.gz".format(p_id)) for p_id in pat_nums]
    all_images_dict = {}
    for idx, abs_filename in enumerate(abs_file_list):
        new_img4d = None
        patient_id = os.path.basename(abs_filename).replace("_4d.nii.gz", "")
        pat_id = int(patient_id.replace("patient", ""))
        img = sitk.ReadImage(abs_filename)
        voxel_spacing4d = img.GetSpacing()[::-1]
        img4d_arr = sitk.GetArrayFromImage(img)
        num_of_frames = img4d_arr.shape[0]
        orig_number_of_frames = img4d_arr.shape[0]
        if patient_id == "patient015" or patient_id == 15 or \
                patient_id == "patient034" or patient_id == 34 or \
                patient_id == "patient045" or patient_id == 45:
            num_of_frames = 20
        for frame_id in range(num_of_frames):
            original_spacing = voxel_spacing4d[1:]
            spacing = voxel_spacing4d[1:]
            img_np = img4d_arr[frame_id]
            num_slices = img_np.shape[0]
            if resample or original_spacing[-1] < 1.:
                img_np = apply_2d_zoom_3d(img_np, spacing, do_blur=False, new_spacing=new_spacing, as_type=np.int, order=1)
                spacing = np.array([original_spacing[0], new_spacing[1], new_spacing[2]]).astype(np.float32)

            new_img4d = np.vstack((new_img4d, np.expand_dims(img_np, axis=0))) if new_img4d is not None else np.expand_dims(img_np, axis=0)

        image_dict = {'image': new_img4d, 'spacing': spacing, 'patient_id': patient_id, "num_frames": num_of_frames,
                    'original_spacing': original_spacing, 'num_slices': num_slices,
                    'orig_num_frames': orig_number_of_frames}

        all_images_dict[pat_id] = image_dict
    return all_images_dict


def acdc_loop4d_slice_dim(root_dir='~/data/ACDC/all_cardiac_phases', fold=None, patid_list=None, resample=False,
                       rescale=False, new_spacing=None, limited_load=False, file_suffix='4d.nii.gz',
                       dataset="validation", as4d=False):
    """
        Basically the same procedure as in ACDC.data acdc_all_image4d.
        But in this procedure we loop over slice dim and a volume contains the same slice for all time-points
    """
    if patid_list is not None:
        if isinstance(patid_list[0], str):
            patid_list = [int(patid.strip('patient')) for patid in patid_list]
        allpatnumbers = patid_list
    elif fold is not None:
        allpatnumbers = get_acdc_patient_ids(fold, dataset, limited_load=False)
    else:
        allpatnumbers = np.arange(1, 101)
    if limited_load:
        allpatnumbers = allpatnumbers[:3]

    root_dir = os.path.expanduser(root_dir)
    if new_spacing is None:
        new_spacing = ACDCImage.new_spacing

    for patnum in allpatnumbers:
        # IMPORTANT resample and rescale in ACDCImage method
        img = ACDCImage(patnum, root_dir=root_dir, resample=resample, scale_intensities=rescale,
                            new_spacing=new_spacing,
                            file_suffix=file_suffix)

        img4d_arr = img.image4d()
        num_of_frames, num_of_slices, _, _ = img4d_arr.shape
        if img.patient_id == "patient015" or img.patient_id == 15 or \
            img.patient_id == "patient034" or img.patient_id == 34 or \
                img.patient_id == "patient045" or img.patient_id == 45:

            print("WARNING - ACDC4D - skipping patient/frames above 20".format(img.patient_id))

        if as4d:
            yield img.preprocessed4d()
        else:
            # Important. Returns dict with numpy 4d pre-processed image (rescaled and resampled if applicable)
            img4d_dict = img.preprocessed4d()
            num_of_slices = img4d_dict['num_slices']
            for slice_id in range(num_of_slices):
                img_np = img4d_dict['image'][:, slice_id]
                yield {'image': img_np, 'spacing': img4d_dict['spacing'], 'reference': img_np,
                       'patient_id': img.patient_id, 'slice_id': slice_id,
                        'cardiac_phase': ' ', 'structures': [], 'original_spacing': img4d_dict['original_spacing']}


def get_acdc_patient_sets(limited_load=False, split_file=None):
    if split_file is None:
        acdc_config = get_config('ACDC')
        src_path = os.path.expanduser(acdc_config.data_root_dir)
        split_file = os.path.join(src_path, "train_val_test_split_sr.yaml")
    pat_nums_training = get_patids_acdc_sr("training", split_file=split_file, limited_load=limited_load)
    pat_nums_training.sort()
    pat_nums_validation = get_patids_acdc_sr("validation", split_file=split_file, limited_load=limited_load)
    pat_nums_validation.sort()
    pat_nums_test = get_patids_acdc_sr("test", split_file=split_file, limited_load=limited_load)
    pat_nums_test.sort()
    return {'training': pat_nums_training, 'validation': pat_nums_validation, 'test': pat_nums_test}
