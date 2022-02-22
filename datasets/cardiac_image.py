import numpy as np
import os
from pathlib import Path
from datasets.common import read_nifty, apply_2d_zoom_3d


def get_cardiac4d_per_frame(src_data_path, rescale=True, resample=False, new_spacing=None,
                          patid_list=None, file_suffix=".nii.gz", resize_perc=tuple((0, 100))):
    filepath_generator = Path(src_data_path).rglob('*' + file_suffix)
    abs_file_list = [file_obj for file_obj in filepath_generator]
    abs_file_list.sort()
    data_dict = {}
    # IMPORTANT: actually the abs_file_list contains for each patient a tuple (filename, patient_idx (integer))
    # we did this because the patient_ids contain alphanumeric characters
    for path_obj in abs_file_list:
        patient_id = path_obj.name.replace(file_suffix, "")
        if patid_list is not None:
            if patient_id not in patid_list:
                continue
        img = CardiacImage(str(path_obj.resolve()), filename_ref_labels=None, rescale=rescale, resample=resample,
                           new_spacing=new_spacing, pat_num=-1, resize_perc=resize_perc)
        for sample in img.all_phases():
            yield sample


def get_cardiac4d(src_data_path, rescale=True, resample=False, new_spacing=None,
                          patid_list=None, file_suffix=".nii.gz", resize_perc=tuple((0, 100))):
    filepath_generator = Path(src_data_path).rglob('*' + file_suffix)
    abs_file_list = [file_obj for file_obj in filepath_generator]
    abs_file_list.sort()
    # IMPORTANT: actually the abs_file_list contains for each patient a tuple (filename, patient_idx (integer))
    # we did this because the patient_ids contain alphanumeric characters
    data_dict = {}

    for path_obj in abs_file_list:
        patient_id = path_obj.name.replace(file_suffix, "")
        if patid_list is not None:
            if patient_id not in patid_list:
                continue

        img = CardiacImage(str(path_obj.resolve()), filename_ref_labels=None, rescale=rescale, resample=resample,
                           new_spacing=new_spacing, pat_num=-1, resize_perc=resize_perc)
        data_dict[patient_id] = img.preprocessed4d()
    return data_dict


class CardiacImage(object):

    new_spacing = np.array([1, 1.4, 1.4])

    def __init__(self, filename, filename_ref_labels=None, rescale=True, resample=False, new_spacing=None,
                 pat_num=-1, resize_perc=tuple((0, 100))):
        # IMPORTANT: ADDED pat_num which is a replacement for the alphanumeric ARVC patient id. This number
        # is actually stored in the split file (filename, pat_num). We store both numbers in the dict below
        self._filename = filename
        self._rescale = rescale
        self._resample = resample
        # read_nifty returns numpy array of shape [#frames, z, y, x]
        self._img, self._spacing, self._direction, self._origin = read_nifty(self._filename, get_extra_info=True)
        self._spacing = np.array(self._spacing[1:]).astype(np.float64)  # skip frame dimension for spacing
        self.original_spacing = self._spacing
        self.number_of_frames, self.number_of_slices, _, _ = self._img.shape
        self._filename_ref_labels = filename_ref_labels
        self.resize_perc = resize_perc

        stem_filename = os.path.splitext(os.path.basename(self._filename))[0]
        self.patient_id = stem_filename.strip('.nii.gz')
        self.pat_num = pat_num
        self.new_spacing = new_spacing
        if new_spacing is None and resample:
            self.new_spacing = CardiacImage.new_spacing

    def all_phases(self):

        for frame_id in range(self.number_of_frames):
            original_spacing = self.original_spacing
            spacing = self._spacing
            img_np = self._img[frame_id]
            num_slices = img_np.shape[0]
            if self._resample:
                img_np = apply_2d_zoom_3d(img_np, spacing, do_blur=True, new_spacing=self.new_spacing)
                spacing = np.array([original_spacing[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)
            min_val, max_val = np.percentile(img_np, self.resize_perc)
            if max_val - min_val == 0:
                print("WARNING - CardiacImage - skipping patient/frame {}/{}".format(self.patient_id, frame_id))
                continue
            if self._rescale:
                img_np = self._rescale_intensities(img_np, percentile=self.resize_perc).astype(np.float32)
            yield {'image': img_np, 'spacing': spacing, 'patient_id': self.patient_id,
                   'frame_id': frame_id, 'origin': self._origin,
                   'direction': self._direction,
                   'original_spacing': self.original_spacing, 'num_slices': num_slices}

    def preprocessed4d(self):

        new_img4d = None
        for frame_id in range(self.number_of_frames):
            original_spacing = self.original_spacing
            spacing = self._spacing
            img_np = self._img[frame_id]
            num_slices = img_np.shape[0]
            if self._resample:
                img_np = apply_2d_zoom_3d(img_np, spacing, do_blur=True, new_spacing=self.new_spacing)
                spacing = np.array([original_spacing[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)
            min_val, max_val = np.percentile(img_np, self.resize_perc)
            if max_val - min_val == 0:
                print("WARNING - CardiacImage - skipping patient/frame {}/{}".format(self.patient_id, frame_id))
                continue
            if self._rescale:
                img_np = self._rescale_intensities(img_np, percentile=self.resize_perc).astype(np.float32)

            new_img4d = np.vstack((new_img4d, np.expand_dims(img_np, axis=0))) if new_img4d is not None else np.expand_dims(img_np, axis=0)
        return {'image': new_img4d, 'spacing': spacing, 'patient_id': self.patient_id, "num_frames": self.number_of_frames,
                'original_spacing': original_spacing, 'num_slices': num_slices, 'pat_num': self.pat_num,
                'origin': self._origin, 'direction': self._direction}

    @staticmethod
    def _rescale_intensities(img_data, percentile=(1, 99)):
        min_val, max_val = np.percentile(img_data, percentile)
        return ((img_data.astype(float) - min_val) / (max_val - min_val)).clip(0, 1)
