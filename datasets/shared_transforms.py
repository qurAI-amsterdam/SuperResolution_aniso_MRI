import torch
import numpy as np
import cv2
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.abstract_transforms import Compose
from torchvision import transforms


def create_new_sample(image, sample):
    new_sample = {'image': image}
    new_sample.update((newkey, newvalue) for newkey, newvalue in sample.items() if newkey not in new_sample.keys())
    return new_sample


class SpatialTransformToHalfBatch(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if not isinstance(output_size, tuple):
            self.output_size = tuple((output_size, output_size))
        else:
            self.output_size = output_size

        spatial_transform = SpatialTransform(self.output_size, None,
                         data_key="data",
                         do_elastic_deform=True, alpha=(0., 1500.), sigma=(30., 50.),
                         do_rotation=True, angle_x=(0, np.pi / 9),
                         do_scale=False,
                         border_mode_data='constant', border_cval_data=0, order_data=1,
                         random_crop=False)
        self.transform1 = transforms.Compose([AdjustToPatchSize(self.output_size), CenterCrop(self.output_size)])
        self.transform2 = Compose([spatial_transform])

    def __call__(self, sample):
        image = sample['image']
        num_dims = image.ndim
        image1, image2 = np.split(image, 2, axis=0)
        temp = {'data': image2[:, None], 'other key': 'other value'}
        image2_trans = self.transform2(**temp)['data']
        image1_trans = self.transform1({'image': image1})['image']
        if num_dims == 3 and image2_trans.ndim == 4:
            image2_trans = image2_trans.squeeze(axis=1)
        new_image = np.concatenate((image1_trans, image2_trans), axis=0)
        new_sample = create_new_sample(new_image, sample)
        return new_sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.

    """

    def __init__(self, output_size, input_padding=None, rs=np.random):
        assert isinstance(output_size, (int, tuple))
        self.rs = rs
        if not isinstance(output_size, tuple):
            self.output_size = tuple((output_size, output_size))
        else:
            self.output_size = output_size
        self.input_padding = input_padding

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """
        image = sample['image']
        if isinstance(image, np.ndarray):
            i_dim = image.ndim
        else:
            # assuming torch tensor
            i_dim = image.dim()
        if i_dim == 4:
            # Only for pytorch tensor
            image = torch.squeeze(image, dim=1)
            _, h, w = image.shape
        elif i_dim == 3:
            _, h, w = image.shape
        else:
            h, w = image.shape
        new_h, new_w = self.output_size
        # the rare case that the output size is equal to image size, we don't do anything
        if new_h == h and new_w == w:
            return sample
        top = self.rs.randint(0, h - new_h)
        left = self.rs.randint(0, w - new_w)

        if self.input_padding:
            new_h += 2*self.input_padding
            new_w += 2*self.input_padding
        if image.ndim == 3:
            """
                Important: We assume shape [z, y, x]
            """
            image = image[:, top:  top + new_h,
                              left: left + new_w]
            if "slice_between" in sample.keys():
                sample["slice_between"] = sample["slice_between"][:, top:  top + new_h, left: left + new_w]
            if "loss_mask" in sample.keys() and sample['loss_mask'].ndim == 3:
                sample["loss_mask"] = sample["loss_mask"][:, top:  top + new_h, left: left + new_w]
        else:
            image = image[top:  top + new_h,
                          left: left + new_w]
            if "slice_between" in sample.keys():
                sample["slice_between"] = sample["slice_between"][top:  top + new_h, left: left + new_w]
            if "loss_mask" in sample.keys() and sample['loss_mask'].ndim == 2:
                sample["loss_mask"] = sample["loss_mask"][top:  top + new_h, left: left + new_w]

        if i_dim == 4:
            # Only for pytorch tensor
            image = torch.unsqueeze(image, dim=1)
        new_sample = create_new_sample(image, sample)

        del sample
        return new_sample


class RandomCropNextToCenter(object):

    def __init__(self, patch_size, max_translation, rs=None, fixed_translation=False):
        self.patch_size = patch_size
        self.max_translation = max_translation
        self.rs = rs
        self.fixed_translation = fixed_translation

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """
        image = sample['image']
        if image.ndim == 3:
            num_slices, w, h = image.shape
        else:
            w, h = image.shape

        w_half, h_half = w // 2, h // 2
        patch_half = self.patch_size // 2
        if w_half - self.max_translation - patch_half < 0:
            if w_half - patch_half <= 0:
                w_range = np.arange(0, 1)
            else:
                max_translation_w = w_half - patch_half
                w_range = np.arange(-max_translation_w, max_translation_w)
        else:
            max_translation_w = self.max_translation
            w_range = np.arange(-max_translation_w, max_translation_w)

        if h_half - self.max_translation - patch_half < 0:
            if h_half - patch_half <= 0:
                h_range = np.arange(0, 1)
            else:
                max_translation_h = h_half - patch_half
                h_range = np.arange(-max_translation_h, max_translation_h)
        else:
            max_translation_h = self.max_translation
            h_range = np.arange(-max_translation_h, max_translation_h)

        if not self.fixed_translation:
            w_choice = self.rs.choice(w_range)
            h_choice = self.rs.choice(h_range)
            w_start, h_start = w_half + w_choice - patch_half, h_half + h_choice - patch_half
        else:
            w_start, h_start = w_half + self.max_translation - patch_half, h_half + self.max_translation - patch_half
        w_pad, h_pad = None, None
        if w_start < 0:
            w_pad = abs(w_start)
            w_start = 0

        if h_start < 0:
            h_pad = abs(h_start)
            h_start = 0

        if image.ndim == 3:
            """
                Important: We assume shape [z, y, x]
            """
            new_image = image[:, w_start:w_start + self.patch_size, h_start:h_start + self.patch_size]
        else:
            new_image = image[w_start:w_start + self.patch_size, h_start:h_start + self.patch_size]

        new_sample = create_new_sample(new_image, sample)

        return new_sample


class RandomAnyRotation(object):
    def __init__(self, max_degree=45, rs=np.random, fixed_degree=None):
        self.max_degree = max_degree
        self.fixed_degree = fixed_degree
        self.rs = rs

    def __call__(self, sample):
        image = sample['image']
        dtype_save = None
        if image.dtype != np.float32 and image.dtype != np.float64:
            dtype_save = image.dtype
            image = image.astype(np.float32)
        if self.fixed_degree is None:
            degree = self.rs.randint(0, self.max_degree)
        else:
            degree = self.fixed_degree
        num_rows, num_cols = image.shape[1:]

        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), degree, 1)
        for i in np.arange(image.shape[0]):
            image[i] = cv2.warpAffine(image[i], rotation_matrix, (num_cols, num_rows))

        if dtype_save is not None:
            image = np.round(image)
        new_sample = create_new_sample(image, sample)

        del sample

        return new_sample


class RandomRotation(object):
    def __init__(self, rs=np.random):
        self.rs = rs

    def __call__(self, sample):
        """

        :param sample:  We assume sample['image'] has shape [z, y, x] or [y, x]
        :return:
        """
        image = sample['image']
        k = self.rs.randint(0, 4)

        if image.ndim == 2:
            new_image = np.rot90(image, k, (0, 1)).copy()
            if "slice_between" in sample.keys():
                sample["slice_between"] = np.rot90(sample["slice_between"], k, (0, 1)).copy()
            if "loss_mask" in sample.keys() and sample['loss_mask'].ndim == 2:
                sample["loss_mask"] = np.rot90(sample["loss_mask"], k, (0, 1)).copy()
        elif image.ndim == 3:
            # [io_channels, x, y]
            new_image = np.rot90(image, k, (1, 2)).copy()
            if "slice_between" in sample.keys():
                sample["slice_between"] = np.rot90(sample["slice_between"], k, (1, 2)).copy()
            if "loss_mask" in sample.keys() and sample['loss_mask'].ndim == 3:
                sample["loss_mask"] = np.rot90(sample["loss_mask"], k, (1, 2)).copy()
        else:
            raise ValueError("ERROR - RandomRotation - image rank not supported")
        new_sample = create_new_sample(new_image, sample)
        del sample
        return new_sample


class GenericToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        new_sample = {}

        try:
            for key in sample.keys():
                if key == 'image' and sample[key].dtype != np.float32 and sample[key].dtype != np.float64:
                    new_sample[key] = sample[key].astype(np.float32)
                if isinstance(sample[key], np.ndarray):
                    new_sample[key] = torch.from_numpy(sample[key])
                elif isinstance(sample[key], np.int32) or isinstance(sample[key], np.int64) \
                        or isinstance(sample[key], np.float) or isinstance(sample[key], int):
                    new_sample[key] = torch.from_numpy(np.array([sample[key]]).astype(type(sample[key])))
                elif isinstance(sample[key], np.float32):
                    new_sample[key] = torch.from_numpy(np.array([sample[key]]).astype(np.float32))
                elif isinstance(sample[key], tuple):
                    new_sample[key] = torch.from_numpy(np.array([sample[key]]).astype(np.float32))
                elif isinstance(sample[key], list):
                    new_sample[key] = torch.from_numpy(np.array([sample[key]]))
                elif isinstance(sample[key], str):
                    new_sample[key] =sample[key]
                else:
                    # Skip all other types for now
                    # new_sample[key] = torch.from_numpy(np.array([sample[key]]).astype(type(sample[key])))
                    print("WARNING - GenericToTensor - Skipping key {}: {}".format(key, type(sample[key])))

        except ValueError:
            for key in sample.keys():

                if key == 'image' and sample[key].dtype != np.float32 and sample[key].dtype != np.float64:
                    new_sample[key] = sample[key].astype(np.float32)
                if isinstance(sample[key], np.ndarray):
                    new_sample[key] = torch.from_numpy(np.ascontiguousarray(sample[key]))

        del sample
        return new_sample


class CenterCrop(object):
    """Center crop image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if not isinstance(output_size, tuple):
            self.output_size = tuple((output_size, output_size))
        else:
            self.output_size = output_size

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """
        image = sample['image']
        if image.ndim == 4:
            _, _, h, w = image.shape
        elif image.ndim == 3:
            _, h, w = image.shape
        else:
            h, w = image.shape
        half_w = int(self.output_size[0] / 2)
        half_h = int(self.output_size[1] / 2)

        img_half_w = int(w / 2)
        img_half_h = int(h / 2)
        slice_w = slice(img_half_w - half_w,
                        img_half_w + half_w, None)
        slice_h = slice(img_half_h - half_h,
                        img_half_h + half_h, None)

        if image.ndim == 4:
            """
                Important: We assume shape [t, z, y, x]
            """
            image = image[:, :, slice_h, slice_w]
            if "slice_between" in sample.keys():
                sample["slice_between"] = sample["slice_between"][:, :, slice_h, slice_w]
            if "loss_mask" in sample.keys() and sample['loss_mask'].ndim == 4:
                sample["loss_mask"] = sample["loss_mask"][:, :, slice_h, slice_w]
        elif image.ndim == 3:
            """
                Important: We assume shape [z, y, x]
            """
            image = image[:, slice_h, slice_w]
            if "slice_between" in sample.keys():
                sample["slice_between"] = sample["slice_between"][:, slice_h, slice_w]
            if "loss_mask" in sample.keys() and sample['loss_mask'].ndim == 3:
                sample["loss_mask"] = sample["loss_mask"][:, slice_h, slice_w]

        else:
            image = image[slice_h, slice_w]
            if "slice_between" in sample.keys():
                sample["slice_between"] = sample["slice_between"][slice_h, slice_w]
            if "loss_mask" in sample.keys() and sample['loss_mask'].ndim == 2:
                sample["loss_mask"] = sample["loss_mask"][slice_h, slice_w]
        new_sample = create_new_sample(image, sample)
        del sample
        return new_sample


class RandomIntensity(object):
    def __init__(self, rs=np.random, slice_mask=None):
        self.rs = rs
        self.slice_mask = slice_mask
        # self.maximum_g = 1.25
        # self.maximum_gain = 10

    def __call__(self, sample):
        image = sample['image']

        gain = self.rs.uniform(2.5, 7.5)
        cutoff = self.rs.uniform(0.25, 0.75)
        if self.slice_mask is None:
            image = (1 / (1 + np.exp(gain * (cutoff - image))))
        else:
            image[self.slice_mask] = (1 / (1 + np.exp(gain * (cutoff - image[self.slice_mask]))))
        if "slice_between" in sample.keys():
            sample["slice_between"] = (1 / (1 + np.exp(gain * (cutoff - sample["slice_between"]))))
        new_sample = create_new_sample(image, sample)
        del sample
        return new_sample


class AdjustToPatchSize(object):

    def __init__(self, patch_size):
        # IMPORTANT!!! although we pass a tuple, currently we assume a quadratic patch size
        self.patch_size = patch_size

    def __call__(self, sample):

        image = sample['image']
        if image.ndim == 3:
            # [z, w, h]
            num_slices, w, h = image.shape
            delta_w_l, delta_w_r, delta_h_l, delta_h_r = self._get_padding(w, h)
            image = np.pad(image,
                          ((0, 0), (delta_w_l, delta_w_r),
                           (delta_h_l, delta_h_r)),
                           'constant',
                           constant_values=(0,)).astype(np.float32)
            if "slice_between" in sample.keys():
                sample["slice_between"] = np.pad(sample["slice_between"],
                          ((0, 0), (delta_w_l, delta_w_r),
                           (delta_h_l, delta_h_r)),
                           'constant',
                           constant_values=(0,)).astype(np.float32)
            if "loss_mask" in sample.keys() and sample['loss_mask'].ndim == 3:
                sample["loss_mask"] = np.pad(sample["loss_mask"],
                                                 ((0, 0), (delta_w_l, delta_w_r),
                                                  (delta_h_l, delta_h_r)),
                                                 'constant',
                                                 constant_values=(0,)).astype(np.float32)
        elif image.ndim == 2:
            w, h = image.shape
            delta_w_l, delta_w_r, delta_h_l, delta_h_r = self._get_padding(w, h)
            image = np.pad(image, ((delta_w_l, delta_w_r),
                           (delta_h_l, delta_h_r)), 'constant',
                           constant_values=(0,)).astype(np.float32)
            if "slice_between" in sample.keys():
                sample["slice_between"] = np.pad(sample["slice_between"], ((delta_w_l, delta_w_r),
                           (delta_h_l, delta_h_r)), 'constant',
                           constant_values=(0,)).astype(np.float32)
            if "loss_mask" in sample.keys() and sample['loss_mask'].ndim == 2:
                sample["loss_mask"] = np.pad(sample["loss_mask"], ((delta_w_l, delta_w_r),
                                                                           (delta_h_l, delta_h_r)), 'constant',
                                                 constant_values=(0,)).astype(np.float32)
        new_sample = create_new_sample(image, sample)
        return new_sample

    def _get_padding(self, w, h):
        delta_w_l, delta_w_r, delta_h_l, delta_h_r = 0, 0, 0, 0
        if w < self.patch_size[0]:
            delta_w = self.patch_size[0] - w
            delta_w_l = delta_w // 2
            delta_w_r = delta_w // 2 if delta_w % 2 == 0 else int(delta_w_l + 1)
        if h < self.patch_size[1]:
            delta_h = self.patch_size[0] - h
            delta_h_l = delta_h // 2
            delta_h_r = delta_h // 2 if delta_h % 2 == 0 else int(delta_h_l + 1)

        return delta_w_l, delta_w_r, delta_h_l, delta_h_r
