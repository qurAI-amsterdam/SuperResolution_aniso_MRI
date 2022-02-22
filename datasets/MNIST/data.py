import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import RandomSampler
import torch
import copy
import cv2
from datasets.ACDC.data import sitk_save


def create_3d_volume(dataset, patid_offset=0, degree_step=10, do_save=True, debug=False, output_dir=None):
    def rotation_any(img, degree=45):
        assert img.ndim == 2
        num_cols, num_rows = img.shape
        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), degree, 1)
        return cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    # assuming dataset is object returned by get_data_loaders_mnist function
    assert 360 % degree_step == 0
    if do_save and output_dir is None:
        raise ValueError("Error - output dir cannot be None!")
    else:
        output_dir = os.path.expanduser(output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=False)
    num_rotations = (360 // degree_step)
    images3d = []
    for idx, (img2d, lbl) in enumerate(dataset):
        img2d = img2d.squeeze().numpy()
        img3d = img2d[None]
        for i in np.arange(1, num_rotations):
            img3d = np.concatenate((img3d, rotation_any(img2d, degree_step * i)[None]), axis=0)
        if do_save:
            fname = os.path.join(output_dir, "{:05d}.nii.gz".format(idx+patid_offset+1))
            sitk_save(fname, img3d, np.array([num_rotations, 1, 1]).astype(np.float64), dtype=np.float32)
        else:
            images3d.append(img3d)
        if debug and idx > 2:
            print("Warning - debug - exiting after 2 iterations!")
            break
    if not do_save:
        return images3d
    else:
        return None


class RandomTranslation(object):
    def __init__(self, patch_size=64, rs=np.random, max_translation=None):
        self.rs = rs
        self.patch_size = patch_size
        self.max_translation = max_translation
        if self.max_translation is not None:
            assert patch_size > max_translation

    def __call__(self, *args):
        new_image = np.zeros((self.patch_size, self.patch_size)).astype(np.uint8)
        image = args[0]
        image = np.array(image).astype(np.uint8)
        w, h = image.shape
        if self.max_translation is not None:
            max_w, max_h = self.max_translation, self.max_translation
        else:
            max_w, max_h = self.patch_size - w, self.patch_size - h

        w_range, h_range = np.arange(0, max_w), np.arange(0, max_h)
        w_start, h_start = self.rs.choice(w_range), self.rs.choice(h_range)
        new_image[w_start:w_start + w, h_start:h_start + h] = image
        return new_image


def get_data_loaders_mnist(batch_size, use_cuda=True, data_dir='~/data/', test_batch_size=16,
                           transform_tr=transforms.Compose([transforms.ToTensor()]),
                           transform_te=None):
    if transform_te is None:
        transform_te = transform_tr
    data_dir = os.path.expanduser(data_dir)
    kwargs = {'num_workers': 2, 'pin_memory': False} if use_cuda else {}
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