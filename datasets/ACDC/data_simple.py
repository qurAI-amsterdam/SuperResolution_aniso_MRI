import numpy as np
import torch
import copy
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from datasets.ACDC.data import get_acdc_patient_ids
from datasets.ACDC.data import read_nifty
import os.path as path
from datasets.common import apply_2d_zoom_3d
from collections import defaultdict
from datasets.common import get_paired_slices, get_leave_one_out_slices, create_acdc_abs_file_list
from datasets.ACDC.acdc_transforms import CropNextToCenter, ToTensor, CenterCrop, RandomAnyRotation, RandomTranslation
from torch.utils.data.sampler import RandomSampler


def get_dataset_acdc_labels(args, dta_settings, rs, acdc_set="both", slice_interpol="random", fixed_loc=None, cardiac_label=None,
                            transform_tr=None, transform_te=None):
    if transform_tr is None:
        transform_tr = transforms.Compose([CropNextToCenter(80, rs=rs),
                                                                         RandomAnyRotation(rs=rs, max_degree=359),
                                                                         RandomTranslation(patch_size=128,
                                                                                           rs=rs,
                                                                                           max_translation=1),
                                                                         ToTensor()])
    if transform_te is None:
        transform_te = transforms.Compose([CropNextToCenter(80, loc=fixed_loc), # args['width']
                                                                    RandomTranslation(patch_size=128,
                                                                                      rs=rs,
                                                                                      max_translation=None,
                                                                                      apply_separately=True),
                                           ToTensor()])
    # cardiac_label in {1: 'RV', 2: 'MYO', 3: 'LV'}
    training_set = None
    val_set = None

    if acdc_set in ['both', 'train']:
        training_set = ACDCLabels('training',
                                           fold=args['fold'],
                                  cardiac_label=cardiac_label,
                                           root_dir=dta_settings.short_axis_dir,
                                           resample=True,
                                           adjacency=slice_interpol,
                                           transform=transform_tr,
                                           limited_load=args['limited_load'])
    if fixed_loc is None:
        loc = 0
    else:
        loc = fixed_loc
    if acdc_set in ['both', 'test']:
        val_set = ACDCLabels('validation',
                                      fold=args['fold'],
                             cardiac_label=cardiac_label,
                                      root_dir=dta_settings.short_axis_dir,
                                      resample=True,
                                      adjacency="paired",
                                      transform=transform_te,
                                      limited_load=False)
    return training_set, val_set


def get_data_loaders_acdc_labels(dataset, args, use_cuda=True):
    loader = None
    kwargs = {'num_workers': 8} if use_cuda else {}
    if dataset.dataset == "training":
        # compute total number of iters for main loop
        # num_of_iters = args.batch_size * len(dataset)
        sampler = RandomSampler(dataset, replacement=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], sampler=sampler, drop_last=True, **kwargs)
    if dataset.dataset == "validation":
        loader = torch.utils.data.DataLoader(dataset, batch_size=args['test_batch_size'], shuffle=False, drop_last=True, **kwargs)

    return loader


class ACDCLabels(Dataset):

    def __init__(self, dataset,
                 fold=0,
                 adjacency="combined",
                 cardiac_label=None,   # {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}
                 root_dir='~/data/ACDC/all_cardiac_phases',
                 transform=None, limited_load=False,
                 rescale=True,
                 resample=False):
        self.dataset = dataset
        self.adjacency = adjacency
        self.cardiac_label = cardiac_label
        self._root_dir = root_dir
        self.transform = transform
        self._resample = resample
        pat_nums = get_acdc_patient_ids(fold, dataset, limited_load=limited_load)
        labels = list()
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
        if self.cardiac_label is None:
            tissue_classes = np.arange(2, 4) # LV and LVM
        else:
            tissue_classes = [self.cardiac_label]

        for idx, number in tqdm(enumerate(pat_nums), desc='Load {} set fold {}'.format(dataset, fold), total=len(pat_nums)):
            patnum = pat_nums[idx]
            self._path = root_dir + '/patient{:03d}'.format(number)
            img = ACDCImageLabel(patnum, root_dir=self._path, resample=self._resample, scale_intensities=rescale, abs_filename=None)

            for cardiac_phase in ['ED', 'ES']:
                if cardiac_phase == "ED":
                    ref_labels, sp = img.ed()
                else:
                    ref_labels, _ = img.es()
                frame_id = img.get_frame_id(cardiac_phase)
                for lbl in tissue_classes:
                    filtered_labels = self._filter_non_slices(ref_labels, f_lbl=lbl)
                    filtered_labels = self._normalize_labels(filtered_labels)
                    labels.append(filtered_labels)
                    num_slices = filtered_labels.shape[0]
                    spacing.append(sp)
                    self.z_spacings[sp[0]].append("patient_id{:03d}".format(patnum))
                    org_spacing.append(copy.deepcopy(sp))
                    frame_ids.append(frame_id)
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
        self._labels = labels
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

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self._idcs)

    def _normalize_labels(self, labels):
        new_labels = np.zeros_like(labels).astype(np.float32)
        new_labels[labels == self.cardiac_label] = 2.
        return new_labels

    def __getitem__(self, idx):

        if self.adjacency == "paired":
            img_idx, slice_idx, slice_idx_interp = self._idcs[idx]
            paired_image = np.vstack((np.expand_dims(self._labels[img_idx][slice_idx], 0),
                                      np.expand_dims(self._labels[img_idx][slice_idx_interp], 0)))
            slice_pair = self._slice_pairs[img_idx][slice_idx]
            vol_slices = self._vol_num_slices[img_idx]
            slice_ids = (slice_pair + 1) / vol_slices
        elif self.adjacency == "combined" or self.adjacency == "volume-combined":
            img_idx, slice_idx, slice_idx_interp, target_slice_idx_interp = self._idcs[idx]
            # print("slice_idx, slice_idx_interp, target_slice_idx_interp ", slice_idx, slice_idx_interp, target_slice_idx_interp)
            paired_image = np.vstack((np.expand_dims(self._labels[img_idx][slice_idx], 0),
                                      np.expand_dims(self._labels[img_idx][slice_idx_interp], 0)))
            slice_pair = self._slice_pairs[img_idx][slice_idx].astype(np.int8)
            compute_aux_loss = self._extra_loss[img_idx][slice_idx]
            if compute_aux_loss == 1:
                target_slice = np.expand_dims(self._labels[img_idx][target_slice_idx_interp], 0)
            else:
                target_slice = np.zeros((1, paired_image.shape[1], paired_image.shape[2])).astype(np.float32)
            vol_slices = self._vol_num_slices[img_idx]
            slice_ids = (slice_pair + 1)[:-1] / vol_slices
        elif self.adjacency == "random":
            img_idx, slice_idx = self._idcs[idx]
            paired_image = np.expand_dims(self._labels[img_idx][slice_idx], axis=0)
            slice_pair = self._slice_pairs[img_idx][slice_idx]
            vol_slices = np.array([self._vol_num_slices[img_idx]]).astype(np.float32)
            slice_ids = slice_pair + 1 / vol_slices
        elif self.adjacency == "volume":
            img_idx, _ = self._idcs[idx]
            paired_image = self._labels[img_idx]
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
        # if self.adjacency == "combined" or self.adjacency == "volume-combined":
        #     if target_slice is not None:
        #         sample['target_slice'] = target_slice
        #     sample['compute_aux_loss'] = np.array([compute_aux_loss]).astype(np.int)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _filter_non_slices(self, labels, f_lbl):

        new_labels = np.zeros_like(labels)
        new_labels[labels == f_lbl] = labels[labels == f_lbl]
        labels = new_labels
        slice_label_counts = np.count_nonzero(labels, axis=(1, 2))
        slice_idx = slice_label_counts != 0
        return labels[slice_idx]

    @staticmethod
    def _create_file_list(list_patids, acdc_root_dir):
        acdc_file_list = create_acdc_abs_file_list(list_patids, acdc_root_dir)
        return acdc_file_list


class ACDCImageLabel(object):
    new_spacing = tuple((1., 1.4, 1.4))

    def __init__(self, number, root_dir=None, scale_intensities=True, resample=False, abs_filename=None):
        # number IS patient id without "patient" prefix, just integer without lpad zeros
        self._number = number
        self._path = root_dir
        self._patient_id = "patient{:03d}".format(number)
        self._scale_intensities = scale_intensities
        # IMPORTANT: resampling of patient029 ends up in different shape then before. Hence, for this patient we do not resample
        if self._number == 29:
            self._resample = False
        else:
            self._resample = resample
        self.spacing = None
        self.original_spacing = None
        self.base_apex_slice_es = None
        self.base_apex_slice_ed = None
        self.frame_id = None

    def ed(self):
        idx = int(self.info()['ED'])
        self.frame_id = idx
        im, self.original_spacing = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}.nii.gz'.format(self._number, idx)),
                                               False)
        self.original_spacing = self._check_spacing(self.original_spacing)
        gt, _ = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}_gt.nii.gz'.format(self._number, idx)),
                           resample_zaxis=False, as_type=np.int)
        self.spacing = self.original_spacing

        if self._resample or self.original_spacing[-1] < 1.:
            gt = self._do_resample(gt, self.original_spacing)
            self.spacing = np.array([self.original_spacing[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)

        self.base_apex_slice_ed = self._determine_apex_base_slices(gt)
        return gt, self.spacing

    def es(self):
        idx = int(self.info()['ES'])
        self.frame_id = idx
        im, self.original_spacing = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}.nii.gz'.format(self._number, idx)),
                                               False)
        self.original_spacing = self._check_spacing(self.original_spacing)
        gt, _ = read_nifty(path.join(self._path, 'patient{:03d}_frame{:02d}_gt.nii.gz'.format(self._number, idx)),
                           resample_zaxis=False, as_type=np.int)
        self.spacing = self.original_spacing if isinstance(self.original_spacing, np.ndarray) else \
            np.array(self.original_spacing)

        if self._resample or self.original_spacing[-1] < 1.:
            gt = self._do_resample(gt, self.original_spacing)
            self.spacing = np.array([self.original_spacing[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)
        self.base_apex_slice_es = self._determine_apex_base_slices(gt)
        return gt, self.spacing

    def get(self, cardiac_phase):
        if cardiac_phase == "ES":
            im, self.spacing, gt = self.es()
            return im, self.spacing, gt, self.frame_id
        else:
            im, self.spacing, gt = self.ed()
            return im, self.spacing, gt, self.frame_id

    @staticmethod
    def _check_spacing(spacing):
        # if not ndarray (e.g. tuple) convert to numpy. Otherwise dataloader gets stuck on tuples (expects np arrays)
        return spacing if isinstance(spacing, np.ndarray) else np.array(spacing).astype(np.float32)

    def get_frame_id(self, cardiac_phase):
        return int(self.info()[cardiac_phase])

    @property
    def patient_id(self):
        return self._patient_id

    def shape(self):
        return self._img4d.header.get_data_shape()[::-1]

    def _do_resample(self, gt_lbl, sp):
        gt_lbl = apply_2d_zoom_3d(gt_lbl, sp, order=0, do_blur=False, as_type=np.int, new_spacing=self.new_spacing)
        return gt_lbl

    @staticmethod
    def _determine_apex_base_slices(labels):
        """

        :param labels: numpy array of shape [z, y, x]
        :return: dict with 'A' = apex and 'B' = base keys. Values of dict are scalar slice ids
        """
        slice_ab = {'A': None, 'B': None}
        # Note: low-slice number => most basal slices / high-slice number => most apex slice
        # Note: assuming labels has one bg-class indicated as 0-label and shape [z, y, x]
        slice_ids = np.arange(labels.shape[0])
        # IMPORTANT: we sum over x, y and than check whether we'have a slice that has ZERO labels. So if
        # np.any() == True, this means there is a slice without labels.
        binary_mask = (np.sum(labels, axis=(1, 2)) == 0).astype(np.bool)
        if np.any(binary_mask):
            # we have slices (apex/base) that do not contain any labels. We assume that this can only happen
            # in the first or last slices e.g. [1, 1, 0, 0, 0, 0] so first 2 slice do not contain any labels
            slices_with_labels = slice_ids[binary_mask != 1]
            slice_ab['B'], slice_ab['A'] = int(min(slices_with_labels)), int(max(slices_with_labels))
        else:
            # all slices contain labels. We simply assume slice-idx=0 --> base and slice-idx = max#slice --> apex
            slice_ab['B'], slice_ab['A'] = int(min(slice_ids)), int(max(slice_ids))
        return slice_ab

    def info(self):
        try:
            self._info
        except AttributeError:
            self._info = dict()
            fname = self._path + '/Info.cfg'
            with open(fname, 'r') as f:
                for l in f:
                    k, v = l.split(':')
                    self._info[k.strip()] = v.strip()
        finally:
            return self._info