import numpy as np
import torch
from torch.nn.modules.utils import _pair
import cv2
from scipy.ndimage import gaussian_filter


def create_new_sample(image, sample):
    new_sample = {'image': image}
    new_sample.update((newkey, newvalue) for newkey, newvalue in sample.items() if newkey not in new_sample.keys())
    return new_sample


class AddRandomNoise(object):
    def __init__(self, rs=np.random):
        self.rs = rs

    def __call__(self, sample):
        image = sample['image']
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32)
        new_image = (image * 255. + self.rs.uniform(size=image.shape)) / 256.
        new_sample = create_new_sample(new_image.astype(np.float32), sample)

        return new_sample


class BlurImage(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        image = sample['image']
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32)

        if image.ndim == 3:
            new_image = np.zeros_like(image)
            for i in range(image.shape[0]):
                new_image[i] = gaussian_filter(image[i], self.sigma)
        elif image.ndim == 2:
            new_image = gaussian_filter(image, self.sigma)
        else:
            raise ValueError("BlurImage - Error - ndim not supported {}".format(image.ndim))
        new_sample = create_new_sample(new_image.astype(np.float32), sample)
        del sample
        return new_sample


class RandomTranslation(object):
    def __init__(self, patch_size=128, rs=np.random, max_translation=None, apply_separately=False, max_distance=False):
        self.rs = rs
        self.patch_size = patch_size
        self.max_translation = max_translation
        self.apply_separately = apply_separately
        self.max_distance = max_distance

    def __call__(self, sample):
        if self.max_distance is None:
            max_distance = False  # True if self.rs.uniform() > 0.5 else False
        else:
            max_distance = self.max_distance

        image = sample['image']
        if image.ndim == 3:
            num_slices, h, w = image.shape
            new_image = np.zeros((num_slices, self.patch_size, self.patch_size)).astype(image.dtype)
        else:
            h, w = image.shape
            new_image = np.zeros((self.patch_size, self.patch_size)).astype(image.dtype)
        if self.max_translation is not None:
            assert (self.patch_size - w) // 2 >= self.max_translation
        if self.max_translation is not None:
            max_w, max_h = self.max_translation, self.max_translation
        else:
            max_w, max_h = (self.patch_size - w) // 2, (self.patch_size - h) // 2

        w_half, h_half = w // 2, h // 2
        patch_half = self.patch_size // 2
        w_range, h_range = np.arange(-max_w, max_w), np.arange(-max_h, max_h)
        if max_distance:
            w_start, h_start = patch_half - w_half + max_w, patch_half - h_half + max_h
        else:
            w_start, h_start = patch_half + self.rs.choice(w_range) - w_half, patch_half + self.rs.choice(h_range) - h_half
        if image.ndim == 3:
            """
                Important: We assume shape [z, y, x]
            """
            if self.apply_separately:
                # first slice
                new_image[0, w_start:w_start + w, h_start:h_start + h] = image[0]
                for i in np.arange(1, new_image.shape[0]):
                    if max_distance:
                        w_start, h_start = patch_half - w_half - max_w, patch_half - h_half - max_h
                    else:
                        w_start, h_start = patch_half + self.rs.choice(w_range) - w_half, patch_half + self.rs.choice(h_range) - h_half
                    new_image[i, w_start:w_start + w, h_start:h_start + h] = image[i]
            else:
                new_image[:, w_start:w_start + w, h_start:h_start + h] = image
        else:
            new_image[w_start:w_start + w, h_start:h_start + h] = image

        new_sample = create_new_sample(new_image, sample)

        return new_sample


class RandomCropNextToCenter(object):

    def __init__(self, patch_size, max_translation, rs=None, fixed_translation=False):
        self.patch_size = patch_size
        self.max_translation = max_translation
        self.rs = rs
        self.fixed_translation = fixed_translation
        self.apply_separately = False

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
        w_range, h_range = np.arange(-self.max_translation, self.max_translation), np.arange(-self.max_translation, self.max_translation)
        if not self.fixed_translation:
            w_choice = self.rs.choice(w_range)
            h_choice = self.rs.choice(h_range)
            w_start, h_start = w_half + w_choice - patch_half, h_half + h_choice - patch_half
        else:
            w_start, h_start = w_half + self.max_translation - patch_half, h_half + self.max_translation - patch_half
        if image.ndim == 3:
            """
                Important: We assume shape [z, y, x]
            """
            if self.apply_separately:
                # first slice
                new_image[0] = image[0, w_start:w_start + w, h_start:h_start + h]
                for i in np.arange(1, new_image.shape[0]):
                    w_start, h_start = w_half + self.rs.choice(w_range) - patch_half, h_half + self.rs.choice(h_range) - patch_half
                    new_image[i] = image[i, w_start:w_start + w, h_start:h_start + h]
            else:
                new_image = image[:, w_start:w_start + self.patch_size, h_start:h_start + self.patch_size]
        else:
            new_image = image[w_start:w_start + self.patch_size, h_start:h_start + self.patch_size]

        # _, w_new, h_new = new_image.shape
        # if w_new != self.patch_size or h_new != self.patch_size:
        #     print("error ", self.patch_size, image.shape, new_image.shape)
        #     print(w_start, w_half, w_choice, patch_half)
        #     print(h_start, h_half, h_choice, patch_half)
        new_sample = create_new_sample(new_image, sample)

        return new_sample


class CropNextToCenter(object):
    """Center crop image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.

    """

    def __init__(self, output_size, loc=None, rs=None):
        assert isinstance(output_size, (int, tuple))
        self.output_size = _pair(output_size)
        self.loc = loc
        self.rs = rs

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

        img_half_w = int(w / 2)
        img_half_h = int(h / 2)
        if self.loc is None:
            loc = self.rs.choice(a=[0, 1, 2, 3, 4])
        else:
            loc = self.loc
        if loc == 0:
            # upper-left from center
            slice_w = slice(img_half_w - self.output_size[0], img_half_w, None)
            slice_h = slice(img_half_h - self.output_size[1], img_half_h, None)
        elif loc == 1:
            # lower-right from center
            slice_w = slice(img_half_w, img_half_w + self.output_size[0], None)
            slice_h = slice(img_half_h, img_half_h + self.output_size[1], None)
        elif loc == 2:
            # upper-right from center
            slice_w = slice(img_half_w, img_half_w + self.output_size[0], None)
            slice_h = slice(img_half_h - self.output_size[1], img_half_h, None)
        elif loc == 3:
            # lower-left from center
            slice_w = slice(img_half_w - self.output_size[0], img_half_w, None)
            slice_h = slice(img_half_h, img_half_h + self.output_size[1], None)
        else:
            # center
            slice_w = slice(img_half_w - self.output_size[0] // 2,
                            img_half_w + self.output_size[0] // 2, None)
            slice_h = slice(img_half_h - self.output_size[1] // 2,
                            img_half_h + self.output_size[1] // 2, None)
        if image.ndim == 3:
            """
                Important: We assume shape [z, y, x]
            """
            image = image[:, slice_h, slice_w]
        else:
            image = image[slice_h, slice_w]
        new_sample = create_new_sample(image, sample)
        return new_sample


class RandomMirroring(object):
    def __init__(self, p=0.5, rs=np.random):
        self.p = p
        self.rs = rs

    def __call__(self, sample):
        image = sample['image']
        rs = self.rs
        if rs.rand() < 0.5:
            if image.ndim == 3:
                axis = (1, 2)
            else:
                axis = (0, 1)
            image = np.flip(image, axis=axis)

        new_sample = create_new_sample(image, sample)
        del sample
        return new_sample


class RandomPerspective(object):
    def __init__(self, rs=np.random):
        self.rs = rs

    def __call__(self, sample):
        image = sample['image']
        M = np.identity(3, float)
        M += self.rs.uniform(-0.002, 0.002, (3, 3))
        shape = (image.shape[-1], image.shape[-2])  # works for 2D and 3D
        if image.ndim == 3:
            image = cv2.warpPerspective(image.transpose(1, 2, 0), M, shape, flags=cv2.INTER_LINEAR).transpose(2, 0, 1)
        else:
            image = cv2.warpPerspective(image, M, shape, flags=cv2.INTER_LINEAR)

        new_sample = create_new_sample(image, sample)

        del sample
        return new_sample


class RandomAnyRotation(object):
    def __init__(self, max_degree=45, rs=np.random):
        self.max_degree = max_degree
        self.rs = rs

    def __call__(self, sample):
        image = sample['image']
        dtype_save = None
        if image.dtype != np.float32 and image.dtype != np.float64:
            dtype_save = image.dtype
            image = image.astype(np.float32)

        degree = self.rs.randint(0, self.max_degree)
        num_rows, num_cols = image.shape[1:]

        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), degree, 1)
        for i in np.arange(image.shape[0]):
            image[i] = cv2.warpAffine(image[i], rotation_matrix, (num_cols, num_rows))

        if dtype_save is not None:
            image = np.round(image)
        new_sample = create_new_sample(image, sample)
        if 'target_slice' in sample.keys() and sample['compute_aux_loss']:
            target_slice = sample['target_slice']
            new_sample['target_slice'] = np.expand_dims(cv2.warpAffine(target_slice[0], rotation_matrix, (num_cols, num_rows)), axis=0)
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


class CenterCrop(object):
    """Center crop image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = _pair(output_size)

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
        if 'target_slice' in sample.keys():
            target_slice = sample['target_slice']
            if target_slice.ndim == 3:
                """
                    Important: We assume shape [z, y, x]
                """
                target_slice = target_slice[:, slice_h, slice_w]
            else:
                target_slice = target_slice[slice_h, slice_w]
            new_sample['target_slice'] = target_slice

        del sample
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
        self.output_size = _pair(output_size)
        self.input_padding = input_padding

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
        new_h, new_w = self.output_size
        rs = self.rs

        top = rs.randint(0, h - new_h)
        left = rs.randint(0, w - new_w)

        if self.input_padding:
            new_h += 2*self.input_padding
            new_w += 2*self.input_padding
        if image.ndim == 3:
            """
                Important: We assume shape [z, y, x]
            """
            image = image[:, top:  top + new_h,
                              left: left + new_w]
        else:
            image = image[top:  top + new_h,
                          left: left + new_w]

        new_sample = create_new_sample(image, sample)
        if 'target_slice' in sample.keys():
            target_slice = sample['target_slice']
            if target_slice.ndim == 3:
                """
                    Important: We assume shape [z, y, x]
                """
                target_slice = target_slice[:, top:  top + new_h,
                                            left: left + new_w]
            else:
                target_slice = target_slice[top:  top + new_h,
                                            left: left + new_w]
            new_sample['target_slice'] = target_slice

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
            image = np.rot90(image, k, (0, 1)).copy()
        elif image.ndim == 3:
            # [io_channels, x, y]
            image = np.rot90(image, k, (1, 2)).copy()
        else:
            raise ValueError("ERROR - RandomRotation - image rank not supported")
        new_sample = create_new_sample(image, sample)
        del sample
        return new_sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alphas']
        spacing, original_spacing = sample['spacing'], sample['original_spacing']
        patient_id, frame_id, num_slices_vol = sample['patient_id'], sample['frame_id'], sample['num_slices_vol']
        slice_id = sample['slice_id']
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32)

        try:
            image = torch.from_numpy(image)
            spacing = torch.from_numpy(spacing)
            original_spacing = torch.from_numpy(original_spacing)
            alpha = torch.from_numpy(alpha)
            patient_id = torch.from_numpy(patient_id)
            frame_id = torch.from_numpy(frame_id)
            num_slices_vol = torch.from_numpy(num_slices_vol)
            slice_id = torch.from_numpy(slice_id)
        except ValueError:
            image = torch.from_numpy(np.ascontiguousarray(image))
            alpha = torch.from_numpy(np.ascontiguousarray(alpha))
            spacing = torch.from_numpy(np.ascontiguousarray(spacing))
            original_spacing = torch.from_numpy(np.ascontiguousarray(original_spacing))
            patient_id = torch.from_numpy(np.ascontiguousarray(patient_id))
            frame_id = torch.from_numpy(np.ascontiguousarray(frame_id))
            num_slices_vol = torch.from_numpy(np.ascontiguousarray(num_slices_vol))
            slice_id = torch.from_numpy(np.ascontiguousarray(slice_id))

        new_sample = {'image': image,
                      'alphas': alpha,
                      'num_slices_vol': num_slices_vol,
                      'slice_id': slice_id,
                      'spacing': spacing,
                      'original_spacing': original_spacing,
                      'patient_id': patient_id,
                      'frame_id': frame_id}
        if 'target_slice' in sample.keys():
            target_slice = sample['target_slice'].copy()
            new_sample['target_slice'] = torch.from_numpy(target_slice)
        if 'compute_aux_loss' in sample.keys():
            new_sample['compute_aux_loss'] = torch.from_numpy(sample['compute_aux_loss'])
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