import numpy as np
from tqdm import tqdm, tqdm_notebook
from datasets.ARVC.dataset import ARVCImage
from datasets.data_config import get_config
from datasets.ARVC.common import get_abs_filenames_segmentation

arvc_data_settings = get_config('ARVC')


def do_limit_load(p_dataset_filelist):
    return p_dataset_filelist[:arvc_data_settings.limited_load_max]


def translate_labels(labels, direction):
    assert direction in ['arvc_to_acdc', 'acdc_to_arvc']
    new_labels = np.zeros_like(labels).astype(np.int)
    if direction == 'arvc_to_acdc':
        new_labels[labels == 1] = arvc_data_settings.cls_translate[1]
        new_labels[labels == 2] = arvc_data_settings.cls_translate[2]
    elif direction == 'acdc_to_arvc':
        new_labels[labels == arvc_data_settings.cls_translate[1]] = 1
        new_labels[labels == arvc_data_settings.cls_translate[2]] = 2
    return new_labels


def get_arvc_4d_edes_image_generator(dataset, limited_load=False, resample=False, rescale=True, patid=None,
                                     new_spacing=None):
    """
    We use this function during validation and testing. Different than for the ARVCDataSet object which returns
    slices that are used for training and hence transformed e.g. rotation, mirroring etc.

    :param dataset:
    :param limited_load:
    :param resample:
    :param rescale:
    :param patid: one patient ID to process (string)
    :param all_frames: boolean, if TRUE, we get all time frames of a patient ignoring reference labels
    :return:
    """
    dta_settings = get_config('ARVC')
    assert dataset in dta_settings.datasets
    files_to_load = get_abs_filenames_segmentation()[dataset]
    if patid is not None:
        files_to_load = [fname for fname in files_to_load if patid in fname]
        if len(files_to_load) == 0:

            raise ValueError("ERROR - {} is not a valid patient id".format(patid))
    if limited_load:
        files_to_load = do_limit_load(files_to_load)
    patient_data_idx = {}

    idx = 0

    for filename in tqdm(files_to_load, desc="Load {} set".format(dataset)):
        filename_ref_labels = filename.replace(dta_settings.short_axis_dir, dta_settings.ref_label_dir)
        img = ARVCImage(filename, filename_ref_labels=filename_ref_labels, rescale=rescale,
                        resample=resample)
        if new_spacing is not None:
            img.new_spacing = new_spacing
        if img.has_labels:
            patient_data_idx[img.patient_id] = []
            for sample_img, sample_lbl in img.ed():
                # IMPORTANT: assuming we use this function to SR ARVC images+labels with a model trained on ACDC
                # Hence, we need to translate label ID from ARVC to ACDC.
                new_labels_ed = np.zeros_like(sample_lbl['labels']).astype(np.int)
                new_labels_ed[sample_lbl['labels'] == 1] = arvc_data_settings.cls_translate[1]
                new_labels_ed[sample_lbl['labels'] == 2] = arvc_data_settings.cls_translate[2]
                sample = {'image': sample_img['image'],
                          'labels': new_labels_ed,
                          'spacing': sample_img['spacing'],
                          'direction': sample_img['direction'],
                          'origin': sample_img['origin'],
                          'frame_id': sample_img['frame_id'],
                          'cardiac_phase': sample_img['cardiac_phase'],
                          'structures': sample_lbl['structures'],
                          'ignore_label': sample_lbl['ignore_label'],
                          'original_spacing': sample_lbl['original_spacing'],
                          'patient_id': sample_lbl['patient_id'],
                          'num_frames': sample_img['number_of_frames'],
                          'orig_num_frames': sample_img['number_of_frames']}
                patient_data_idx[img.patient_id].append(idx)
                yield sample
                idx += 1

            for sample_img, sample_lbl in img.es():
                new_labels_es = np.zeros_like(sample_lbl['labels']).astype(np.int)
                new_labels_es[sample_lbl['labels'] == 1] = arvc_data_settings.cls_translate[1]
                new_labels_es[sample_lbl['labels'] == 2] = arvc_data_settings.cls_translate[2]
                sample = {'image': sample_img['image'],
                          'labels': new_labels_es,
                          'spacing': sample_img['spacing'],
                          'direction': sample_img['direction'],
                          'origin': sample_img['origin'],
                          'frame_id': sample_img['frame_id'],
                          'cardiac_phase': sample_img['cardiac_phase'],
                          'structures': sample_lbl['structures'],
                          'ignore_label': sample_lbl['ignore_label'],
                          'original_spacing': sample_lbl['original_spacing'],
                          'patient_id': sample_lbl['patient_id'],
                          'num_frames': sample_img['number_of_frames'],
                          'orig_num_frames': sample_img['number_of_frames']}
                patient_data_idx[img.patient_id].append(idx)
                yield sample
                idx += 1