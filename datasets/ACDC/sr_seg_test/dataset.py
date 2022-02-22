import os
import glob
from os import path
import numpy as np
from datasets.ACDC.data import read_nifty, rescale_intensities, get_acdc_patient_ids
from datasets.data_config import get_config
from tqdm import tqdm, tqdm_notebook
import SimpleITK as sitk
import nibabel as nib
from kwatsch.common import isnotebook
from datasets.common import apply_2d_zoom_3d
from torch.utils.data import Dataset


DATA_SETTINGS = get_config('ACDCTESTSR')


def acdctestsr_validation_fold(fold, root_dir=DATA_SETTINGS.data_root_dir, limited_load=False, resample=False, patid=None):
    # patid --> should be integer, so 21 instead of patient021
    rescale = True
    if isinstance(patid, str):
        patid = int(patid.strip('patient'))

    allpatnumbers = np.arange(1, 101)
    foldmask = np.tile(np.arange(4)[::-1].repeat(5), 5)
    validation_nums = allpatnumbers[foldmask == fold]
    if patid is not None:
        validation_nums = [pid for pid in validation_nums if patid == pid]
        if len(validation_nums) == 0:
            raise ValueError("ERROR - acdc validation fold - patid {} not found in validation set".format(patid))
    if limited_load:
        validation_nums = validation_nums[:3]

    for patnum in validation_nums:
        img = ACDCTESTSRImage(patnum, root_dir=root_dir, resample=resample, rescale=rescale)
        ed_dict = img.ed()
        es_dict = img.es()
        ed, es, ed_gt, es_gt = ed_dict['image'], es_dict['image'], ed_dict['reference'], es_dict['reference']
        sp = ed_dict['spacing']
        if len(sp) > 3:
            sp = sp[1:]
        # STRONG ASSUMPTION FOR SR EVALUATION OF ACDCSR WE HAVE 2 PHASES. ED=0 ES=1 !!!!
        frame_id_ed = img.get_frame_id('ED')
        frame_id_es = img.get_frame_id('ES')

        yield {'image': ed, 'spacing': sp, 'reference': ed_gt, 'patient_id': img.patient_id, 'frame_id': frame_id_ed,
               'cardiac_phase': 'ED', 'structures': [], 'original_spacing': img.original_spacing,
               'direction': img.direction, 'origin': img.origin}
        yield {'image': es, 'spacing': sp, 'reference': es_gt, 'patient_id': img.patient_id, 'frame_id': frame_id_es,
               'cardiac_phase': 'ES', 'structures': [], 'original_spacing': img.original_spacing,
               'direction': img.direction, 'origin': img.origin}


def get_abs_filenames_acdc_sr(fold, dataset, limited_load=False, file_suffix=".nii.gz",
                              image_path=DATA_SETTINGS.short_axis_dir):
    pat_nums = get_acdc_patient_ids(fold, dataset, limited_load=limited_load)
    return [path.join(image_path, "patient{:03d}{}".format(patid, file_suffix)) for patid in pat_nums]


class ACDCTESTSRDataset(Dataset):

    def __init__(self, dataset,
                 fold=0,
                 root_dir=DATA_SETTINGS.data_root_dir,
                 transform=None, limited_load=False,
                 rescale=False,
                 resample=False):

        self._root_dir = root_dir
        self.transform = transform
        self._resample = resample
        pat_nums = get_acdc_patient_ids(fold, dataset, limited_load=limited_load)
        print("INFO Loading ACDCTESTSRDataset - len({}) = {}".format(dataset, len(pat_nums)))
        images = list()
        spacing = list()
        org_spacing = list()
        ids = list()
        cardiac_phases = list()
        frame_ids = list()
        patient_ids = list()
        allidcs = np.empty((0, 2), dtype=int)
        avg_shape = np.zeros(3)
        if isnotebook():
            myiterator = tqdm_notebook(enumerate(pat_nums), desc='Load {} set fold {}'.format(dataset, fold))
        else:
            myiterator = tqdm(enumerate(pat_nums), desc='Load {} set fold {}'.format(dataset, fold))
        for idx, patnum in myiterator:
            img = ACDCTESTSRImage(patnum, root_dir=root_dir, resample=self._resample, rescale=rescale)
            ed_dict = img.ed()
            es_dict = img.es()
            ed, es, ed_gt, es_gt = ed_dict['image'], es_dict['image'], ed_dict['reference'], es_dict['reference']
            sp = ed_dict['spacing']
            frame_id_ed = img.get_frame_id('ED')
            frame_id_es = img.get_frame_id('ES')

            images.append(ed)
            images.append(es)

            spacing.append(sp)
            spacing.append(sp)
            org_spacing.append(img.original_spacing)
            org_spacing.append(img.original_spacing)

            cardiac_phases.append('ED')
            cardiac_phases.append('ES')
            frame_ids.append(frame_id_ed)
            frame_ids.append(frame_id_es)

            ids.append('{:03d} ED'.format(patnum))
            ids.append('{:03d} ES'.format(patnum))
            patient_ids.append('patient{:03d}'.format(patnum))
            patient_ids.append('patient{:03d}'.format(patnum))

            img_idx = (idx * 2)
            allidcs = np.vstack((allidcs, np.vstack((np.ones(len(ed)) * img_idx, np.arange(len(ed)))).T))
            img_idx = (idx * 2) + 1
            allidcs = np.vstack((allidcs, np.vstack((np.ones(len(es)) * img_idx, np.arange(len(es)))).T))

            avg_shape += es.shape
        avg_shape *= 1./(len(images) / 2)

        self._idcs = allidcs.astype(int)
        self._images = images
        self._spacings = spacing
        self._org_spacings = org_spacing
        self._cardiac_phases = cardiac_phases
        self._frame_ids = frame_ids
        self._ids = ids
        self._patient_ids = patient_ids

    def __len__(self):
        return len(self._idcs)

    def __getitem__(self, idx):

        img_idx, slice_idx = self._idcs[idx]

        sample = {'image': self._images[img_idx][slice_idx],
                  'spacing': self._spacings[img_idx],
                  'cardiac_phase': self._cardiac_phases[img_idx],
                  'frame_id': self._frame_ids[img_idx],
                  'patient_id': self._patient_ids[img_idx],
                  'original_spacing': self._org_spacings[img_idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ACDCTESTSRImage(object):
    new_spacing = tuple((1., 1.4, 1.4))

    def __init__(self, number, root_dir=DATA_SETTINGS.data_root_dir, rescale=False, resample=False,
                 file_suffix=DATA_SETTINGS.img_file_ext, with_mask=True):
        # number IS patient id without "patient" prefix, just integer without lpad zeros
        self._patid = number
        self.with_mask = with_mask
        self._patient_id = "patient{:03d}".format(number)
        self.info_path = DATA_SETTINGS.info_path + '/patient{:03d}'.format(number)
        self.image_path = path.join(DATA_SETTINGS.short_axis_dir, 'patient{:03d}{}'.format(self._patid, file_suffix))
        if DATA_SETTINGS.ref_label_dir is not None:
            self.has_labels = True
            self.label_path = path.join(DATA_SETTINGS.ref_label_dir, 'patient{:03d}{}'.format(self._patid, file_suffix))
        else:
            self.has_labels = False
            self.label_path = None
        self.number_of_frames = None
        self._rescale = rescale
        self.file_suffix = file_suffix
        # IMPORTANT: resampling of patient029 ends up in different shape then before.
        # Hence, for this patient we do not resample
        if self._patid == 29:
            self._resample = False
        else:
            self._resample = resample

        self.spacing = None
        self.original_spacing = None
        self.origin, self.direction = None, None
        self.frame_id = None
        self._get_image()

    def _get_image(self):
        # both have shape [#frames, #slices, y, x] ED=0, ES=1
        self.im, self.original_spacing, self.origin, self.direction = read_nifty(self.image_path, get_info=True)
        if self.label_path is not None:
            self.lbl, self.original_spacing = read_nifty(self.label_path, as_type=np.int)
        self.original_spacing = self.original_spacing[1:]

    def ed(self):
        im = self.im[self.get_frame_id('ED')]
        gt = self.lbl[self.get_frame_id('ED')] if self.has_labels else None
        return self._process(im, gt)

    def es(self):
        im = self.im[self.get_frame_id('ES')]
        gt = self.lbl[self.get_frame_id('ES')] if self.has_labels else None
        return self._process(im, gt)

    def _process(self, im, gt):
        self.spacing = self.original_spacing
        if self._rescale:
            im = rescale_intensities(im).astype(np.float32)
        if self._resample or self.original_spacing[-1] < 1.:
            im, gt = self._do_resample(im, gt, self.original_spacing)
            self.spacing = np.array([self.original_spacing[0], self.new_spacing[1], self.new_spacing[2]]).astype(np.float32)

        return {'image': im, 'spacing': self.spacing, 'reference': gt}

    @staticmethod
    def _check_spacing(spacing):
        # if not ndarray (e.g. tuple) convert to numpy. Otherwise dataloader gets stuck on tuples (expects np arrays)
        return spacing if isinstance(spacing, np.ndarray) else np.array(spacing).astype(np.float32)

    def get_frame_id(self, cardiac_phase):
        return int(self.info()[cardiac_phase])

    @property
    def patient_id(self):
        return self._patient_id

    def voxel_spacing(self):
        return np.array(self._img4d.header.get_zooms()[::-1]).astype(np.float32)

    def shape(self):
        return self._img.header.get_data_shape()[::-1]

    def _do_resample(self, img, gt_lbl, sp):
        img = apply_2d_zoom_3d(img, sp, do_blur=True, new_spacing=self.new_spacing)
        if self.has_labels:
            gt_lbl = apply_2d_zoom_3d(gt_lbl, sp, order=0, do_blur=False, as_type=np.int, new_spacing=self.new_spacing)
        return img, gt_lbl

    def info(self):
        try:
            self._info
        except AttributeError:
            self._info = dict()
            fname = self.info_path + '/Info.cfg'
            with open(fname, 'r') as f:
                for l in f:
                    k, v = l.split(':')
                    self._info[k.strip()] = v.strip()
        finally:
            return self._info


if __name__ == '__main__':
    # dataset = ACDCTESTSRDataset('training', fold=2, limited_load=True)
    for img_dict in acdctestsr_validation_fold(fold=0):
        print(img_dict['patient_id'], img_dict['frame_id'], img_dict['image'].shape)