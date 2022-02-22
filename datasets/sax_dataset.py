import os
import numpy as np
import SimpleITK as sitk
from datasets.common import apply_2d_zoom_3d, rescale_intensities, read_nifty
import nibabel as nib
from datasets.data_config import get_config


def get_sax_images_gen(dataset_name, patid_list, resample=False,
                        rescale=False, new_spacing=None, file_suffix='.nii.gz',
                        as4d=False):
    dta_settings = get_config(dataset_name)
    for patnum in patid_list:
        if as4d:
            img = SAXImage(patnum, dta_settings, resample=resample, scale_intensities=rescale,
                           new_spacing=new_spacing,
                           file_suffix=file_suffix)
        else:
            # IMPORTANT: we resample and rescale below if necessary!
            img = SAXImage(patnum, dta_settings, resample=False, scale_intensities=False, new_spacing=None,
                           file_suffix=file_suffix)
        img4d_arr = img.array4d()
        num_of_frames = img4d_arr.shape[0]
        orig_num_frames = img4d_arr.shape[0]
        origin = img.origin()
        direction = img.direction()

        if as4d:
            yield img.preprocessed4d()
        else:
            for frame_id in range(num_of_frames):
                img_np = img4d_arr[frame_id]
                spacing = img.spacing()[1:]
                original_spacing = img.spacing()[1:]
                if resample:
                    img_np = apply_2d_zoom_3d(img_np, spacing, do_blur=True, new_spacing=new_spacing)
                    spacing = np.array([original_spacing[0], new_spacing[1], new_spacing[2]]).astype(np.float32)
                if rescale:
                    img_np = rescale_intensities(img_np, percs=(0, 100)).astype(np.float32)

                yield {'image': img_np, 'spacing': spacing, 'reference': img_np, 'patient_id': img.patient_id,
                       'direction': direction, 'origin': origin,
                       'frame_id': frame_id, 'num_frames':  num_of_frames, 'orig_num_frames': orig_num_frames,
                       'cardiac_phase': ' ', 'structures': [], 'original_spacing': original_spacing}


class SAXImage(object):

    def __init__(self, patid, dta_settings, scale_intensities=True, resample=False, rescale_percs=(0, 100),
                 abs_filename=None, new_spacing=None, file_suffix='.nii.gz'):
        assert (resample and new_spacing is not None) or (~resample)
        self._path = dta_settings.short_axis_dir
        self._patient_id = patid
        if abs_filename is None:
            self._img_fname = self._path + os.sep + '{}{}'.format(self._patient_id, file_suffix)
        else:
            self._img_fname = abs_filename
        self._img4d = None
        self._scale_intensities = scale_intensities
        self._rescale_percs = rescale_percs
        self._resample = resample
        self.original_spacing = None
        self._direction, self._origin = None, None
        self.frame_id = None
        self.new_spacing = new_spacing
        self._load()

    def all_phases(self):
        img4d_arr = self.image4d()
        num_of_frames = img4d_arr.shape[0]
        for frame_id in range(num_of_frames):
            self.spacing = self.voxel_spacing4d()[1:]
            self.original_spacing = self.voxel_spacing4d()[1:]
            img_np = img4d_arr[frame_id]
            if self._resample:
                img_np = apply_2d_zoom_3d(img_np, self.spacing, do_blur=True, new_spacing=self.new_spacing)
                self.spacing = np.array([self.original_spacing[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)

            if self._scale_intensities:
                img_np = rescale_intensities(img_np).astype(np.float32)

            yield {'image': img_np, 'spacing': self.spacing, 'patient_id': self.patient_id, 'frame_id': frame_id,
                   'original_spacing': self.original_spacing, 'num_slices': img_np.shape[0]}

    def preprocessed4d(self):
        img4d_arr = self.image4d()
        num_of_frames = img4d_arr.shape[0]
        orig_number_of_frames = img4d_arr.shape[0]

        new_img4d = None
        for frame_id in range(num_of_frames):
            self.original_spacing = self.voxel_spacing4d()[1:]
            self.spacing = self.voxel_spacing4d()[1:]
            img_np = img4d_arr[frame_id]
            num_slices = img_np.shape[0]
            if self._resample or self.original_spacing[-1] < 1.:
                img_np = apply_2d_zoom_3d(img_np, self.spacing, do_blur=True, new_spacing=self.new_spacing)
                self.spacing = np.array([self.original_spacing[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)
            min_val, max_val = np.percentile(img_np, self._rescale_percs)
            if max_val - min_val == 0:
                print("WARNING - ACDC4D - skipping patient/frame {}/{}".format(self.patient_id, frame_id))
                continue
            if self._scale_intensities:
                img_np = rescale_intensities(img_np, percs=self._rescale_percs).astype(np.float32)

            new_img4d = np.vstack((new_img4d, np.expand_dims(img_np, axis=0))) if new_img4d is not None else np.expand_dims(img_np, axis=0)

        return {'image': new_img4d, 'spacing': self.spacing, 'patient_id': self.patient_id, "num_frames": num_of_frames,
                'original_spacing': self.original_spacing, 'num_slices': num_slices,
                'orig_num_frames': orig_number_of_frames}

    @staticmethod
    def _check_spacing(spacing):
        # if not ndarray (e.g. tuple) convert to numpy. Otherwise dataloader gets stuck on tuples (expects np arrays)
        return spacing if isinstance(spacing, np.ndarray) else np.array(spacing).astype(np.float32)

    def get_frame_id(self, cardiac_phase):
        return int(self.info()[cardiac_phase])

    @property
    def patient_id(self):
        return self._patient_id

    def spacing(self):
        return np.array(self._spacing).astype(np.float32)

    def _load(self):
        # self._img4d = nib.load(self._img_fname)
        self._img4d, self._spacing, self._direction, self._origin = read_nifty(self._img_fname, get_extra_info=True)

    def array4d(self):
        return self._img4d

    def shape(self):
        return self._img4d.shape

    def origin(self):
        return self._origin

    def direction(self):
        return self._direction
