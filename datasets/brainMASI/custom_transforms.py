import torch
import numpy as np
import copy


def create_new_sample(image, sample):
    new_sample = {'image': image}
    new_sample.update((newkey, newvalue) for newkey, newvalue in sample.items() if newkey not in new_sample.keys())
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
            new_image = np.zeros((num_slices, self.patch_size, self.patch_size)).astype(image.dtype)
        else:
            w, h = image.shape
            new_image = np.zeros((self.patch_size, self.patch_size)).astype(image.dtype)

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
        elif image.ndim == 3:
            # [io_channels, x, y]
            new_image = np.rot90(image, k, (1, 2)).copy()
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
                elif isinstance(sample[key], tuple):
                    new_sample[key] = torch.from_numpy(np.array([sample[key]]).astype(np.float32))
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

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """
        image = sample['image']

        if image.ndim == 3:
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

        if image.ndim == 3:
            """
                Important: We assume shape [z, y, x]
            """
            image = image[:, slice_h, slice_w]
        else:
            image = image[slice_h, slice_w]
        new_sample = create_new_sample(image, sample)
        del sample
        return new_sample


class RandomIntensity(object):
    def __init__(self, rs=np.random):
        self.rs = rs
        # self.maximum_g = 1.25
        # self.maximum_gain = 10

    def __call__(self, sample):
        image = sample['image']

        gain = self.rs.uniform(2.5, 7.5)
        cutoff = self.rs.uniform(0.25, 0.75)
        image = (1 / (1 + np.exp(gain * (cutoff - image))))

        new_sample = create_new_sample(image, sample)
        del sample
        return new_sample


class AdjustToPatchSize(object):

    def __init__(self, patch_size):
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
        elif image.ndim == 2:
            w, h = image.shape
            delta_w_l, delta_w_r, delta_h_l, delta_h_r = self._get_padding(w, h)
            image = np.pad(image, ((delta_w_l, delta_w_r),
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
