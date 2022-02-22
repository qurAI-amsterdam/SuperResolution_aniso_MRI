import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from datasets.common_brains import get_images
from datasets.common import get_random_adjacent_slice


def get_mnist_ids(dataset):
    assert dataset in ['training', 'validation', 'test']
    print("INFO - get_mnist_ids - {}".format(dataset))
    if dataset == "training":
        return np.arange(1, 60001)
    elif dataset == 'test':
        return np.arange(60001, 60101)
    else:
        return np.arange(69000, 69129)


def get_dataset_MNIST3D(args, src_path, type_of_set="both",
                        transform_tr=None, transform_te=None, test_limited_load=True, downsample=True,
                        downsample_steps=3):
    src_path = os.path.expanduser(src_path)
    training_set = None
    val_set = None

    if type_of_set in ['both', 'train']:
        training_set = MNIST3D('training',
                               root_dir=src_path,
                               rescale=False, resample=False,
                               transform=transform_tr,
                               limited_load=args['limited_load'],
                               slice_selection=args['slice_selection'],
                               downsample=downsample, downsample_steps=downsample_steps)
    if type_of_set in ['both', 'test', 'validation']:
        type_of_set = 'validation' if type_of_set == 'both' else type_of_set
        val_set = MNIST3D(type_of_set, root_dir=src_path,
                          rescale=False, resample=False,
                          transform=transform_te,
                          limited_load=test_limited_load,
                          slice_selection=args['slice_selection'],
                          downsample=downsample, downsample_steps=downsample_steps)
    return training_set, val_set


class MNIST3D(Dataset):

    num_slices = 180

    def __init__(self, dataset,  # Training Test
                 images=None,
                 root_dir='~/data/MNIST3D/',
                 transform=None, limited_load=False,
                 rescale=False,
                 resample=False,
                 rs=np.random.RandomState(1234),
                 slice_selection="adjacent_plus",  # "adjacent_plus"   "mix"
                 downsample=True,
                 downsample_steps=2):
        assert slice_selection in ['adjacent', 'adjacent_plus', 'mix', 'random']
        assert dataset in ['training', 'test', 'validation']
        self._root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self._resample = resample
        self.dataset = dataset
        self.rs = rs
        self.slice_selection = slice_selection
        self.downsample = downsample
        self.downsample_steps = downsample_steps

        if images is None:
            patid_list = get_mnist_ids(dataset)
            print("WARNING - MNIST3D dataset {} ({} vols) - downsample {} "
                  "with factor {} ({})".format(dataset, len(patid_list), downsample, downsample_steps,
                                               self.slice_selection))
            self.images = get_images(patid_list, "MNIST3D", limited_load=limited_load, rescale_int=rescale,
                                     do_downsample=downsample, downsample_steps=downsample_steps,
                                     include_hr_images=False)
        else:
            self.images = images
        self._get_indices()

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

    def _get_slices(self, patnum, slice_id_1, num_slices):
        slice_step = self._get_slice_step()
        slice_id_2 = get_random_adjacent_slice(slice_id_1, num_slices, rs=self.rs, step=slice_step)
        inbetween_slice_id, is_inbetween = self._get_inbetween_sliceid(slice_id_1, slice_id_2)
        if self.rs.choice([0, 1]) == 0:
            slice_idx_from, slice_idx_to = slice_id_1, slice_id_2
        else:
            slice_idx_from, slice_idx_to = slice_id_2, slice_id_1
        img = np.vstack((self.images[patnum]['image'][slice_idx_from][None],
                         self.images[patnum]['image'][slice_idx_to][None],
                         self.images[patnum]['image'][inbetween_slice_id][None]))
        return img, slice_idx_from, slice_idx_to, inbetween_slice_id, is_inbetween

    def __getitem__(self, idx):
        patnum, slice_id_1, num_slices = self._idcs[idx]
        slice_id_1, num_slices = int(slice_id_1), int(num_slices)
        img, slice_idx_from, slice_idx_to, inbetween_slice_id, is_inbetween = \
                self._get_slices(patnum, slice_id_1, num_slices)

        alpha_from, alpha_to = 0.5, 0.5
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