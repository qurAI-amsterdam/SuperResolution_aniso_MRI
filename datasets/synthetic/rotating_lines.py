import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data.sampler import RandomSampler


class RandomAnyRotation(object):
    def __init__(self, max_degree=360, rs=np.random):
        self.max_degree = max_degree
        self.rs = rs

    def __call__(self, sample):
        image = sample['image']
        degree = self.rs.uniform(0, self.max_degree, size=1)[0]
        num_rows, num_cols = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), degree, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
        return {'image': rotated_image, 'degree': np.array([degree])}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, degree = np.expand_dims(sample['image'], axis=0), sample['degree']

        try:
            image = torch.FloatTensor(image, device='cpu')
            degree = torch.Tensor(degree, device='cpu')
        except ValueError:
            # image = torch.from_numpy(np.ascontiguousarray(image)).float()
            # reference = torch.from_numpy(np.ascontiguousarray(reference)).long()
            image = torch.FloatTensor(np.ascontiguousarray(image), device='cpu')
            degree = torch.Tensor(degree, device='cpu')

        return {'image': image, 'degree': degree}


class RotatingLines(Dataset):

    def __init__(self, dataset="training", patch_size=32, transform=transforms.Compose([ToTensor()]), dataset_size=1):
        self.transform = transform
        self.dataset_size = dataset_size
        self.dataset = dataset
        num_cols = patch_size
        num_rows = patch_size
        len_line = patch_size // 2
        x_start, x_stop = len_line // 2, len_line + (len_line // 2)
        # line thickness, currently 2 pixels
        y_start, y_stop = (patch_size // 2) - 1, (patch_size // 2) + 1
        # make default images, a lying bar
        self.img = np.zeros((num_cols, num_rows))
        self.img[x_start:x_stop, y_start:y_stop] = 1
        if dataset == "training":
            self.img = np.expand_dims(self.img, axis=0)
            self.degree = np.array([0])
        else:
            images = np.empty((0, patch_size, patch_size))
            self.degree = np.empty((0, 1))
            for i in np.arange(0, 360):
                rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), i, 1)
                rotated_image = cv2.warpAffine(self.img, rotation_matrix, (num_cols, num_rows))
                images = np.vstack((images, np.expand_dims(rotated_image, axis=0)))
                self.degree = np.vstack((self.degree, np.array([[i]])))
            self.img = images
            self.dataset_size = self.img.shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.dataset == "training":
            img = self.img[0]
            degree = self.degree[0]
        else:
            img = self.img[idx]
            degree = self.degree[idx]
        sample = {'image': img, 'degree': degree}
        if self.transform:
            sample = self.transform(sample)

        return sample



