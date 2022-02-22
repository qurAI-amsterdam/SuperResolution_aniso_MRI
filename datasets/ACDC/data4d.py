import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets.ACDC.data import ACDCImage, get_acdc_patient_ids
from datasets.ACDC.acdc_transforms import RandomCrop, RandomRotation, ToTensor, CenterCrop, RandomAnyRotation, RandomPerspective
from torchvision import transforms
from torch.utils.data.sampler import RandomSampler
from collections import defaultdict
from datasets.common import get_arvc_datasets, create_acdc_abs_file_list, extract_ids, get_paired_slices, get_leave_one_out_slices
from datasets.common import get_paired_frames


def prepare_batch_pairs(batch_dict, expand_type="repeat", dirnet_tester=None, rs=np.random.RandomState(1234)):
    """

    :param batch_dict: batch_dict['image'] has shape [batch, 2, h, w]
            channel 0 = patient slice i
            channel 1 = patient slice i+1 (or i -1)
    :param device: 'cuda' or 'cpu'
    :param expand_type: 'repeat' or 'reshape' or 'split'
    :param dirnet_tester: Bob's DirNet2d tester object. If not None, we use it to register adjacent slices
    :param rs: random state object from numpy
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
        if dirnet_tester is not None:
            raise NotImplementedError("ERROR - this functionality was removed 02/10/2020 (new repo)")

        if expand_type == "split":
            batch_dict['image'], batch_dict['image_b'] = a, b
            batch_dict['slice_id_b'] = batch_dict['slice_id'][..., 1]
            batch_dict['slice_id'] = batch_dict['slice_id'][..., 0]

        else:
            batch_dict['image'] = torch.cat([a, b], dim=0)
            # batch_dict['slice_pair'] = torch.cat([batch_dict['slice_pair'], batch_dict['slice_pair']])
            # slice_id contains tensor of [b, 2]
            batch_dict['slice_id'] = torch.cat([batch_dict['slice_id'][..., 0], batch_dict['slice_id'][..., 1]])
            batch_dict['num_slices_vol'] = torch.cat([batch_dict['num_slices_vol'], batch_dict['num_slices_vol']])
            batch_dict['spacing'] = torch.cat([batch_dict['spacing'], batch_dict['spacing']])
            batch_dict['original_spacing'] = torch.cat([batch_dict['original_spacing'], batch_dict['original_spacing']])
            batch_dict['frame_id'] = torch.cat([batch_dict['frame_id'], batch_dict['frame_id']])
            batch_dict['patient_id'] = torch.cat([batch_dict['patient_id'], batch_dict['patient_id']])
            if "compute_aux_loss" in batch_dict.keys():
                batch_dict['compute_aux_loss'] = torch.cat([batch_dict['compute_aux_loss'], batch_dict['compute_aux_loss']])
            if "target_slice" in batch_dict.keys():
                batch_dict['target_slice'] = torch.cat([batch_dict['target_slice'], batch_dict['target_slice']])
            if "alphas" in batch_dict.keys():
                batch_dict['alphas'] = torch.cat([batch_dict['alphas'], batch_dict['alphas']])

    elif expand_type == "reshape":
        raise NotImplementedError
    elif expand_type == 'single':
        # we will only take the first slice of each batch item
        # Get first item in dim1 (image has shape [b, 2, x, y]
        batch_dict['image'] = batch_dict['image'][:, 0, None]
        # Get first item in dim1 (slice_id has shape [b, #slices]
        batch_dict['slice_id'] = batch_dict['slice_id'][:, 0]
        # Shuffle rows
        batch_dict['image'] = batch_dict['image'][torch.randperm(batch_dict['image'].size(0))]
        if "alphas" in batch_dict.keys():
            batch_dict['alphas'] = torch.rand(batch_dict['image'].size(0)) / 2
    else:
        raise ValueError("Error - prepare_batch_pairs - valid values for expand_type parameter are repeat, reshape, split.")
    return batch_dict


def get_dataset_acdc(args, dta_settings, rs, acdc_set="both", slice_interpol="random"):
    training_set = None
    val_set = None
    if acdc_set in ['both', 'train']:
        training_set = ACDCDataset4DPaired('training',
                                           fold=args.fold,
                                           root_dir=dta_settings.short_axis_dir,
                                           resample=args.resample,
                                           adjacency=slice_interpol,
                                           # RandomCrop(patch_size, None, rs)
                                           transform=transforms.Compose([CenterCrop(args.patch_size),
                                                                         RandomAnyRotation(rs=rs, max_degree=180),
                                                                         RandomPerspective(rs),
                                                                         ToTensor()]),
                                           # RandomRotation(axes=(1, 2), rs=rs)
                                           # transform=transforms.Compose([RandomIntensity(rs), RandomPerspective(rs),
                                           #                               RandomAnyRotation(rs=rs, max_degree=180),
                                           #                               RandomMirroring(axis=1, p=0.5, rs=rs),
                                           #                               RandomCrop(patch_size, None, rs),
                                           #                               ToTensor()]),
                                           limited_load=args.limited_load,
                                           include_arvc=args.include_arvc)
    if acdc_set in ['both', 'test']:
        val_set = ACDCDataset4DPaired('validation',
                                      fold=args.fold,
                                      root_dir=dta_settings.short_axis_dir,
                                      resample=args.resample,
                                      adjacency="paired",
                                      transform=transforms.Compose([CenterCrop(args.patch_size),
                                                                    ToTensor()]),
                                      limited_load=args.limited_load,
                                      include_arvc=False)
    return training_set, val_set


def create_file_list(list_patids, acdc_root_dir, include_arvc=False, limited_load=False):
    acdc_file_list = create_acdc_abs_file_list(list_patids, acdc_root_dir)
    if include_arvc:
        arvc_file_lists = get_arvc_datasets()['training']
        if limited_load:
            arvc_file_lists = arvc_file_lists[:2]
        return acdc_file_list + arvc_file_lists
    else:
        return acdc_file_list


def get_data_loaders_acdc(dataset, args, use_cuda=True):
    loader = None
    kwargs = {'num_workers': 8} if use_cuda else {}
    if dataset.dataset == "training":
        # compute total number of iters for main loop
        # num_of_iters = args.batch_size * len(dataset)
        sampler = RandomSampler(dataset, replacement=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True, **kwargs)
    if dataset.dataset == "validation":
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=True, **kwargs)
    # print("INFO - loader - ", len(loader.sampler), loader.batch_size, len(loader.sampler) / loader.batch_size)
    return loader


class ACDCDataset4DPaired(Dataset):

    """
            Parameter adjacency:

            Random in this context means that "paired_image" only contains one image slice and not TWO which is the case in
            "paired" and "combined" mode. The Pytorch dataloader will create batches with size 1 in dimension 1, whereas in the
            other modes it will return size 2 in dimension 1. We use the function "prepare_batch_pairs" to reshuffle the batches
            so that slices for interpolation will be generated by splitting the batch into two halves. Hence, in those cases the
            batch size doubles to batch_size * 2. In random mode the batch size stays the same and for interpolation the batch is
            just splitted.
    """

    def __init__(self, dataset,  # ['training', 'validation', 'full']
                 fold=0,
                 # (1) paired=slices next to each other,  (2) random=random slices,
                 # (3) combined=random, adjacent, step-over (4) None
                 adjacency="paired",
                 root_dir='~/data/ACDC/all_cardiac_phases',
                 transform=None, limited_load=False,
                 rescale=True,
                 resample=False,
                 include_arvc=False):

        self._root_dir = root_dir
        self.adjacency = adjacency
        self.transform = transform
        self._resample = resample
        self.dataset = dataset
        pat_nums = get_acdc_patient_ids(fold, dataset, limited_load=limited_load)
        print("INFO - ACDCDataset4DPaired - loading {} set - #patients = {} - mode {}".format(dataset, len(pat_nums), self.adjacency))
        images = list()
        spacing = list()
        org_spacing = list()
        slice_pairs = list()
        frame_ids = list()
        patient_ids = list()
        extra_loss = list()
        alphas = list()
        vol_num_slices = list()
        self.z_spacings = defaultdict(list)

        if self.adjacency == "paired":
            allidcs = np.empty((0, 3), dtype=int)
        elif self.adjacency == "random" or self.adjacency is None or self.adjacency == 'volume':
            allidcs = np.empty((0, 2), dtype=int)
        elif self.adjacency == "combined" or self.adjacency == "volume-combined":
            allidcs = np.empty((0, 4), dtype=int)
            # extra_loss = np.empty((0,), dtype=int)
        img_nbr = 0
        abs_file_list = create_file_list(pat_nums, root_dir, include_arvc=include_arvc, limited_load=limited_load)
        if include_arvc:
            pat_nums = list(pat_nums) + list(np.arange(101, 101 + len(abs_file_list) - len(pat_nums)))

        for idx, abs_filename in tqdm(enumerate(abs_file_list), desc='Load {} set fold {}'.format(dataset, fold), total=len(pat_nums)):
            patnum = pat_nums[idx]
            img = ACDCImage(patnum, root_dir=root_dir, resample=self._resample, scale_intensities=rescale, abs_filename=abs_filename)
            for phase_id, data_dict in enumerate(img.all_phases()):
                images.append(data_dict['image'])
                num_slices = data_dict['num_slices']
                spacing.append(data_dict['spacing'])
                self.z_spacings[data_dict['spacing'][0]].append("patient_id{:03d}".format(patnum))
                org_spacing.append(data_dict['original_spacing'])
                frame_ids.append(data_dict['frame_id'])
                patient_ids.append(patnum)
                vol_num_slices.append(num_slices)
                # constructing an array of shape [num_images * num_slices, 3]. axis 1 positions: 0=image num, 1=slice num,
                # 2=slice to interpolate with for training Adv-VAE
                if self.adjacency == "paired":
                    interp_alphas = np.random.uniform(0, 1, size=1).astype(np.float32) / 2
                    alphas.append(interp_alphas.astype(np.float32))
                    slices_from, slices_to = get_paired_slices(num_slices)
                    slice_pairs.append(np.vstack((slices_from, slices_to)).T)
                    allidcs = np.vstack((allidcs, np.vstack((np.ones(num_slices) * img_nbr, slices_from, slices_to)).T))
                elif self.adjacency == "volume-combined":
                    slices_from, slices_to, targetid_interp = get_leave_one_out_slices(num_slices)
                    # target slice id for interpolation. we skip the in between slices
                    compute_aux_loss = np.ones(num_slices).astype(int)
                    interp_alphas = np.array([0.5]).astype(np.float32)
                    alphas.append(interp_alphas)
                    extra_loss.append(compute_aux_loss)
                    slice_pairs.append(np.vstack((slices_from, slices_to, targetid_interp)).T)
                    allidcs = np.vstack((allidcs, np.vstack((np.ones(num_slices) * img_nbr, slices_from, slices_to,
                                                             targetid_interp)).T))
                elif self.adjacency == "combined":
                    # set default values for computation of additional MSE loss for interpolated image
                    compute_aux_loss = np.zeros(num_slices).astype(int)
                    c = np.random.randint(0, high=11, size=1)[0]
                    if 0 <= c <= 4:
                        # adjacent slices
                        interp_alphas = np.random.uniform(0, 1, size=1).astype(np.float32) / 2
                        slices_from, slices_to = get_paired_slices(num_slices)
                        targetid_interp = np.zeros(num_slices)
                    elif 5 <= c <= 7:
                        # from and to slice and target slice are the same
                        interp_alphas = np.array([0.]).astype(np.float32)
                        targetid_interp = np.arange(num_slices)
                        slices_from, slices_to = targetid_interp, targetid_interp
                        compute_aux_loss = np.ones(num_slices).astype(int)
                    elif 8 <= c <= 10:
                        # combined adjacent or skip-one inbetween slice
                        slices_from, slices_to, targetid_interp = get_leave_one_out_slices(num_slices)
                        # target slice id for interpolation. we skip the in between slices
                        compute_aux_loss = np.ones(num_slices).astype(int)
                        interp_alphas = np.array([0.5]).astype(np.float32)
                    # extra_loss = np.vstack((extra_loss, np.vstack((np.ones(num_slices) * img_nbr, compute_aux_loss)).T))
                    alphas.append(interp_alphas.astype(np.float32))
                    extra_loss.append(compute_aux_loss)
                    slice_pairs.append(np.vstack((slices_from, slices_to, targetid_interp.astype(np.float32))).T)
                    allidcs = np.vstack((allidcs, np.vstack((np.ones(num_slices).astype(np.float32) * img_nbr, slices_from, slices_to,
                                                            targetid_interp.astype(np.float32))).T))
                else:
                    # IMPORTANT: self.adjacency NONE or RANDOM
                    # we generate ONE volume for each patient/time frame. This sliceid_interp is just a dummy array with range of slices
                    slice_pairs.append(np.arange(0, num_slices).astype(np.float32))
                    allidcs = np.vstack((allidcs, np.vstack((np.ones(num_slices).astype(np.float32) * img_nbr,
                                                             np.arange(num_slices).astype(np.float32))).T))
                    interp_alphas = np.random.uniform(0, 1, size=1).astype(np.float32) / 2
                    alphas.append(interp_alphas.astype(np.float32))
                img_nbr += 1
        self._idcs = allidcs.astype(int)
        self._images = images
        self._spacings = spacing
        self._org_spacings = org_spacing
        self._frame_ids = frame_ids
        self._slice_pairs = slice_pairs
        self._patient_ids = patient_ids
        self._extra_loss = extra_loss
        self._alphas = alphas
        self._vol_num_slices = vol_num_slices

        print("INFO - Length ACDCDataset4DPaired {}-dataset is {}".format(dataset, len(self)))
        print("INFO - ACDCDataset4DPaired - unique z-spacings ", self.z_spacings.keys())

    def __len__(self):
        return len(self._idcs)

    def __getitem__(self, idx):
        compute_aux_loss = 0
        target_slice = None

        if self.adjacency == "paired":
            img_idx, slice_idx, slice_idx_interp = self._idcs[idx]
            paired_image = np.vstack((np.expand_dims(self._images[img_idx][slice_idx], 0),
                                      np.expand_dims(self._images[img_idx][slice_idx_interp], 0)))
            slice_pair = self._slice_pairs[img_idx][slice_idx]
            vol_slices = self._vol_num_slices[img_idx]
            slice_ids = (slice_pair + 1) / vol_slices
        elif self.adjacency == "combined" or self.adjacency == "volume-combined":
            img_idx, slice_idx, slice_idx_interp, target_slice_idx_interp = self._idcs[idx]
            paired_image = np.vstack((np.expand_dims(self._images[img_idx][slice_idx], 0),
                                      np.expand_dims(self._images[img_idx][slice_idx_interp], 0)))
            slice_pair = self._slice_pairs[img_idx][slice_idx].astype(np.int8)
            compute_aux_loss = self._extra_loss[img_idx][slice_idx]
            if compute_aux_loss == 1:
                target_slice = np.expand_dims(self._images[img_idx][target_slice_idx_interp], 0)
            else:
                target_slice = np.zeros((1, paired_image.shape[1], paired_image.shape[2])).astype(np.float32)
            vol_slices = self._vol_num_slices[img_idx]
            slice_ids = (slice_pair + 1)[:-1] / vol_slices
        elif self.adjacency == "random":
            img_idx, slice_idx = self._idcs[idx]
            paired_image = np.expand_dims(self._images[img_idx][slice_idx], axis=0)
            slice_pair = self._slice_pairs[img_idx][slice_idx]
            vol_slices = np.array([self._vol_num_slices[img_idx]]).astype(np.float32)
            slice_ids = slice_pair + 1 / vol_slices
        elif self.adjacency == "volume":
            img_idx, _ = self._idcs[idx]
            paired_image = self._images[img_idx]
            slice_pair = self._slice_pairs[img_idx]
            vol_slices = self._vol_num_slices[img_idx]
            slice_ids = (slice_pair + 1) / vol_slices
        else:
            raise ValueError("ACDCDataset4DPaired - Error - adjacency parameter only takes [paired, combined, random, volume]")

        alphas = self._alphas[img_idx]
        vol_slices = self._vol_num_slices[img_idx]

        sample = {'image': paired_image,
                  'num_slices_vol': np.array([vol_slices]).astype(np.int),
                  'slice_id': slice_ids.astype(np.float32),
                  'alphas': alphas.astype(np.float32),
                  'spacing': self._spacings[img_idx],
                  'frame_id': np.array([self._frame_ids[img_idx]]).astype(np.int),
                  'patient_id': np.array([self._patient_ids[img_idx]]).astype(np.int),
                  'original_spacing': self._org_spacings[img_idx]}
        if self.adjacency == "combined" or self.adjacency == "volume-combined":
            pass
            # if target_slice is not None:
            #   sample['target_slice'] = target_slice
            # sample['compute_aux_loss'] = np.array([compute_aux_loss]).astype(np.int)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def set_transform(self, transform):
        self.transform = transform


if __name__ == "__main__":
    from datasets.data_config import get_config

    rs = np.random.RandomState(78346)
    dta_settings = get_config("ACDC")
    dataset = ACDCDataset4DPaired('training',
                                   fold=0,
                                   root_dir=dta_settings.short_axis_dir,
                                   resample=False,
                                   transform=transforms.Compose([RandomCrop((32, 32), None, rs), RandomRotation(axes=(1, 2), rs=rs),
                                                                 ToTensor()]),
                                   limited_load=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    print("Lenght dataset ", len(loader.dataset))
    for idx, data_dict in enumerate(loader):
        data_dict = prepare_batch_pairs(data_dict)
        # print(data_dict['patient_id'], data_dict['frame_id'], data_dict['slice_pair'])
        if idx == 10:
            break