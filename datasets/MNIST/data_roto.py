import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets as tv_datasets
from torchvision import transforms
from datasets.MNIST.data3d import MNIST3D, get_mnist_ids


def rotation_any(img, degree=45):
    assert img.ndim == 2
    num_cols, num_rows = img.shape
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), degree, 1)

    return cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))


class MakeRotatedTriple(object):

    def __init__(self, grad_step=5, offset_range=(0, 360), rs=np.random.RandomState(1234)):
        self.rs = rs
        self.grad_step = grad_step
        self.offset_range = offset_range
        self.virtual_slice_id = None

    def __call__(self, slice2d):
        rot_degree_1 = self.rs.randint(self.offset_range[0], self.offset_range[1] - self.grad_step * 2)
        self.virtual_slice_id = rot_degree_1 // self.grad_step
        if self.rs.uniform(0, 1) < 0.5:
            return np.concatenate([rotation_any(slice2d, rot_degree_1)[None],
                                   rotation_any(slice2d, rot_degree_1 + self.grad_step * 2)[None],
                                   rotation_any(slice2d, rot_degree_1 + self.grad_step)[None]])
        else:
            return np.concatenate([rotation_any(slice2d, rot_degree_1 + self.grad_step * 2)[None],
                                   rotation_any(slice2d, rot_degree_1)[None],
                                   rotation_any(slice2d, rot_degree_1 + self.grad_step)[None]])


def get_dataset_MNISTRoto(args, src_path, type_of_set="both",
                          transform_tr=None, transform_te=None, test_limited_load=True, downsample=True,
                          downsample_steps=3):
    src_path = os.path.expanduser(src_path)
    training_set = None
    val_set = None

    if type_of_set in ['both', 'train']:
        training_set = MNISTRoto('training',
                               root_dir=src_path,
                               rescale=False, resample=False,
                               transform=transform_tr,
                               limited_load=args['limited_load'],
                               slice_selection='triplet',
                               downsample=downsample, downsample_steps=downsample_steps)
    if type_of_set in ['both', 'test', 'validation']:
        type_of_set = 'validation' if type_of_set == 'both' else type_of_set
        val_set = MNIST3D(type_of_set, root_dir=os.path.join(src_path, 'MNIST3D'),
                          rescale=False, resample=False,
                          transform=transform_te,
                          limited_load=test_limited_load,
                          slice_selection='adjacent_plus',
                          downsample=downsample, downsample_steps=downsample_steps)
    return training_set, val_set


class MNISTRoto(Dataset):

    num_slices = 180

    def __init__(self, dataset,  # Training Test
                 images=None,
                 root_dir='~/data/MNIST/',
                 transform=None, limited_load=False,
                 rescale=False,
                 resample=False,
                 rs=np.random.RandomState(1234),
                 slice_selection="triplet",
                 downsample=True,
                 downsample_steps=None):
        assert slice_selection in ['triplet']
        assert dataset in ['training']
        assert downsample_steps < 45
        self._root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self._resample = resample
        self.dataset = dataset
        self.limited_load = limited_load
        self.rs = rs
        self.slice_selection = slice_selection
        self.downsample = downsample
        self.downsample_steps = downsample_steps
        self.grad_steps = 15
        self.num_slices = 360 // self.grad_steps
        self.make_triplet = MakeRotatedTriple(self.grad_steps, rs=rs)
        patid_list = get_mnist_ids(dataset)
        print("WARNING - MNISTRoto dataset {} ({} vols) - downsample {} - grad steps {} "
                  "with factor {} ({})".format(dataset, len(patid_list), downsample, self.grad_steps, downsample_steps,
                                               self.slice_selection))
        mnist_transform = transforms.Compose([transforms.ToTensor()])
        print("MNISTroto dataset - self._root_dir {}".format(self._root_dir))
        self.dataset_mnist = tv_datasets.MNIST(self._root_dir, train=True, download=True, transform=mnist_transform)

    def __len__(self):
        if self.limited_load:
            return 1000
        else:
            return len(self.dataset_mnist)

    def __getitem__(self, idx):
        mnist_img, _ = self.dataset_mnist[idx]
        img = self.make_triplet(mnist_img.squeeze().numpy())
        alpha_from, alpha_to = 0.5, 0.5
        sample = {'image': img, 'patient_id': idx, 'num_slices_vol': self.num_slices,
                  'slice_idx_from': self.make_triplet.virtual_slice_id,
                  'slice_idx_to': self.make_triplet.virtual_slice_id + 2,
                  'alpha_from': np.array([alpha_from]).astype(np.float32),
                  'alpha_to': np.array([alpha_to]).astype(np.float32),
                  'inbetween_slice_id': self.make_triplet.virtual_slice_id + 1,
                  'is_inbetween': np.float32(1)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def set_transform(self, transform):
        self.transform = transform
