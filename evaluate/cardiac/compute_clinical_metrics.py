import pandas as pd
import os
import copy
import numpy as np
import yaml
from pathlib import Path
import SimpleITK as sitk
from collections import defaultdict
from tqdm import tqdm_notebook
import cv2


PATH_TO_REFERENCE_LABLES = os.path.expanduser("~/data/ARVC/annotations/contour_ref/")
LABEL_IDS_ARVC = {1: 'LV', 2: 'RV'}
ACDC_TRANSLATE = {1: 3, 2: 1}  # for ACDC LV=3  and RV=1


def retrieve_phase_volume(volume_dict, phase, c_structure):
    # volume_dict: object returned by "get_volumes_all_phases" (see below)
    # but then for a specific patient

    ph_mask = volume_dict['phase_indicators'][c_structure][phase].astype(np.bool)
    vol = volume_dict['volumes'][c_structure]
    return np.argmax(ph_mask.astype(np.int)), vol[ph_mask]


def get_nifti_masks(path_to_auto_labels, pat_list=None):
    path_to_auto_labels = os.path.expanduser(path_to_auto_labels)
    fnames_auto = Path(path_to_auto_labels).glob('*.nii.gz')
    auto_volumes = defaultdict(dict)
    for i, fname in enumerate(fnames_auto):
        patid = fname.resolve().name.replace(".nii.gz", "")
        if "_ni" in patid:
            patid = "_".join(patid.split("_")[:-1])
        if pat_list is not None:
            if patid not in pat_list:
                continue
        img = sitk.ReadImage(str(fname))
        single_voxel_volume = np.product(img.GetSpacing()[:3])
        arr_auto = sitk.GetArrayFromImage(img)
        auto_volumes[patid] = {'labels': arr_auto, 'spacing': np.array(img.GetSpacing()[::-1]).astype(np.float32),
                               'origin': img.GetOrigin(), 'direction': img.GetDirection(),
                               'single_voxel_volume': single_voxel_volume}

    return auto_volumes


def get_patient_data(filename="~/expers/ARVC/arvc_clinical_data.xlsx", exclude=False):
    filename = os.path.expanduser(filename)
    patient_data = pd.read_excel(filename, index_col=False)
    if exclude:
        return patient_data.loc[patient_data.Exclude != 1]
    else:
        return patient_data


def get_cycle_info(path_to_reference=PATH_TO_REFERENCE_LABLES, pat_list=None):
    path_names = Path(path_to_reference).glob('*.nii.gz')
    cycle_info = {}
    for i, filename_ref_labels in enumerate(path_names):
        file_name_info = str(filename_ref_labels.resolve()).replace(".nii.gz", ".yml")
        patid = filename_ref_labels.resolve().name.replace(".nii.gz", "")
        if pat_list is not None:
            if patid not in pat_list:
                continue
        cycle_info[patid] = get_cycle_details(file_name_info)
    return cycle_info


def get_cycle_details(filename):
    """
        filename: absolute filename to reference yaml file (should be in same dir as ref labels)
        loads yaml data for patient details wrt ED/ES phase
    """

    info = None
    if os.path.isfile(filename):
        with open(filename, 'r') as fp:
            info = yaml.load(fp, Loader=yaml.FullLoader)
    return info


def compute_volume_from_mask_via_contours(mask):
    # input: binary segmentation 3D. Returns Binary segmentations converted back from contours
    areas = []
    for m_slice in mask:
        cntrs, hierarchy = cv2.findContours(m_slice.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(cntrs) != 0:
            areas.append(cv2.contourArea(cntrs[0].astype(np.float32)))
    return np.array(areas)


def get_phase_volume(c_arr, info_dict, phase, single_voxel_volume, is_acdc_lbls=False):
    """
        IMPORTANT:
        is_acdc_lbls: must be set to True in case the model was trained on ACDC
    """
    phase_dict = info_dict[phase]
    myvolumes = {}
    for ph, labels in phase_dict.items():
        # LV, RV labels are split over two different time points
        for lbl_id in labels:
            # lbl_id for ARVC is 1=LV or 2=RV, but volumes scored with ACDC trained model
            # label LV=3 and RV=1
            str_lbl_id = 'LV' if lbl_id == 1 else 'RV'
            if is_acdc_lbls:
                lbl_id = ACDC_TRANSLATE[lbl_id]
            # new coding 23-11-2020
            mask = c_arr[ph] == lbl_id
            areas = compute_volume_from_mask_via_contours(mask)
            myvolumes[str_lbl_id] = np.sum(areas * single_voxel_volume) / 1000
            # 23-11-2020: Disabled previous computation. We now first convert masks back to contours
            # and then compute volumes to eliminate bias when using masks (over segmentation)
            # myvolumes[str_lbl_id] = np.sum(c_arr[ph] == lbl_id) * single_voxel_volume / 1000
            # print(str_lbl_id, myvolumes[str_lbl_id])
    return myvolumes


def generate_phase_indicator(info_dict, num_frames, patient_id=None):
    phase_ind = {"LV": {"ED": np.zeros(num_frames).astype(np.int),
                        "ES": np.zeros(num_frames).astype(np.int)},
                 "RV": {"ED": np.zeros(num_frames).astype(np.int),
                        "ES": np.zeros(num_frames).astype(np.int)} }
    for prop_key, property in info_dict.items():
        if prop_key not in ['ED', 'ES']:
            continue
        # ph is ED/ES, phase_dict has int-key time-frames and then list with structure labels e.g. [1, 2] for LV, RV
        for frame_id, labels in property.items():
            for lbl_id in labels:
                str_lbl = LABEL_IDS_ARVC[lbl_id]
                phase_ind[str_lbl][prop_key][frame_id] = 1
    # check
    for struc, phase_dict in phase_ind.items():
        for ph, vec in phase_dict.items():
            # print("{}-{} {}".format(struc, ph, np.where(vec==1)))
            if np.count_nonzero(phase_ind[struc][ph]) != 1:
                print("WARNING - generate_phase_indicator - PatID {}: {}-{} no time frame indication".format(patient_id, struc, ph))
                print(info_dict)

    return phase_ind


def get_volumes_all_phases(auto_masks, cycle_info=None, is_acdc_lbls=False):  # c_arr, info_dict, single_voxel_volume, is_acdc_lbls=False):
    """
        IMPORTANT:
        is_acdc_lbls: must be set to True in case the model was trained on ACDC
    """

    volumes_per_timepoint = dict()
    for p_id, data_dict in auto_masks.items():
        c_arr = data_dict['labels']
        single_voxel_volume = data_dict['single_voxel_volume']
        if cycle_info is not None:
            info_dict = cycle_info[p_id]
            # return dict(dict) key1=LV/RV key2=ED/ES values binary vector indicating ED/ES time frame for structure
            struc_phase_ind = generate_phase_indicator(info_dict, num_frames, patient_id=p_id)
        else:
            struc_phase_ind = None
        num_frames = c_arr.shape[0]

        myvolumes = defaultdict(list)
        for ph in np.arange(num_frames):
            ph_arr = c_arr[ph]
            # LV, RV labels are split over two different time points
            for lbl_id, str_lbl_id in LABEL_IDS_ARVC.items():
                # lbl_id for ARVC is 1=LV or 2=RV, but volumes scored with ACDC trained model
                # label LV=3 and RV=1
                if is_acdc_lbls:
                    lbl_id = ACDC_TRANSLATE[lbl_id]
                mask = ph_arr == lbl_id
                areas = compute_volume_from_mask_via_contours(mask)
                myvolumes[str_lbl_id].append(float(np.sum(areas) * single_voxel_volume / 1000))
        # convert to numpy array
        for k, vec in myvolumes.items():
            myvolumes[k] = np.array(vec).astype(np.float32)

        volumes_per_timepoint[p_id] = {'volumes': myvolumes, 'phase_indicators': struc_phase_ind}

    return volumes_per_timepoint


def get_auto_masks(path_to_auto_labels, pat_list=None, file_suffix=".nii.gz"):
    path_to_auto_labels = os.path.expanduser(path_to_auto_labels)
    fnames_auto = Path(path_to_auto_labels).glob('*' + file_suffix)
    auto_volumes = defaultdict(dict)
    for i, fname in enumerate(fnames_auto):
        patid = fname.resolve().name.replace(file_suffix, "")
        if pat_list is not None:
            if patid not in pat_list:
                continue
        img = sitk.ReadImage(str(fname))
        single_voxel_volume = np.product(img.GetSpacing()[:3])
        arr_auto = sitk.GetArrayFromImage(img)
        auto_volumes[patid] = {'labels': arr_auto, 'spacing': np.array(img.GetSpacing()[::-1]).astype(np.float32),
                               'origin': img.GetOrigin(), 'direction': img.GetDirection(),
                               'single_voxel_volume': single_voxel_volume}

    return auto_volumes


def compute_phase_volumes(auto_labels, cycle_info, is_acdc_lbls=True):
    """
        auto_labels: generated by get_auto_masks
        cycle_info: generated by get_cycle_info
    """
    auto_volumes = defaultdict(dict)
    for patid, data_dict in auto_labels.items():
        arr_auto, single_voxel_volume = data_dict['labels'], data_dict['single_voxel_volume']
        arr_auto = arr_auto[:, arr_auto.any((0, 2, 3))]
        info_auto = cycle_info[patid]
        ed_volumes = get_phase_volume(arr_auto, info_auto, 'ED', single_voxel_volume,
                                                     is_acdc_lbls=is_acdc_lbls)
        es_volumes = get_phase_volume(arr_auto, info_auto, 'ES', single_voxel_volume,
                                                     is_acdc_lbls=is_acdc_lbls)
        if 'LV' not in ed_volumes.keys() or 'LV' not in es_volumes.keys():
            print("WARNING - {}: No LV labels present in auto segmentations".format(patid))
            continue
        if 'RV' not in ed_volumes.keys() or 'RV' not in es_volumes.keys():
            print("WARNING - {}: No RV labels present in auto segmentations".format(patid))
            continue
        auto_volumes[patid]['LV'] = {'EDV': ed_volumes['LV'], 'ESV': es_volumes['LV']}
        auto_volumes[patid]['RV'] = {'EDV': ed_volumes['RV'], 'ESV': es_volumes['RV']}
        auto_volumes[patid]['LV']['SV'] = auto_volumes[patid]['LV']['EDV'] - auto_volumes[patid]['LV']['ESV']
        auto_volumes[patid]['RV']['SV'] = auto_volumes[patid]['RV']['EDV'] - auto_volumes[patid]['RV']['ESV']
        ef_lv = 100 * (auto_volumes[patid]['LV']['EDV'] - auto_volumes[patid]['LV']['ESV']) / auto_volumes[patid]['LV']['EDV']
        ef_rv = 100 * (auto_volumes[patid]['RV']['EDV'] - auto_volumes[patid]['RV']['ESV']) / auto_volumes[patid]['RV']['EDV']

        auto_volumes[patid]['LV']['EF'], auto_volumes[patid]['RV']['EF'] = ef_lv, ef_rv

    return auto_volumes


def peak_filling_rate(volumes, es_ed_indicator_vec, p_id=None):

    diff_lv = -1 * (volumes['LV'][:-1] - volumes['LV'][1:])
    lv_pfr_rt, lv_pfr_tp = np.max(diff_lv), np.argmax(diff_lv)
    if np.argmax(es_ed_indicator_vec['LV']['ES']) > lv_pfr_tp:
        print("Warning - {}: LV - PFR - tp is lower than ES {} < {} (rt={:.2f})".format(p_id, lv_pfr_tp,
                                                                                        np.argmax(es_ed_indicator_vec['LV']['ES']),
                                                                                        lv_pfr_rt))
    # peak_rates[]
    diff_rv = -1 * (volumes['RV'][:-1] - volumes['RV'][1:])
    rv_pfr_rt, rv_pfr_tp = np.max(diff_rv), np.argmax(diff_rv)
    if np.argmax(es_ed_indicator_vec['RV']['ES']) > rv_pfr_tp:
        print("Warning - {}: RV - PFR - tp is lower than ES {} < {} (rt={:.2f})".format(p_id, rv_pfr_tp,
                                                                                        np.argmax(es_ed_indicator_vec['RV']['ES']),
                                                                                        rv_pfr_rt))

    return lv_pfr_rt, rv_pfr_rt


def compute_peak_rates(volumes_over_time):
    """


    :param volumes_over_time: generated with get_volumes_all_phases
    :param es_ed_indicator_vec  generated with get_volumes_all_phases
    :return:
    """

    peak_rates = defaultdict(dict)
    for p_id in volumes_over_time.keys():
        volumes = volumes_over_time[p_id]['volumes']
        es_ed_indicator_vec = volumes_over_time[p_id]['phase_indicators']
        # Peak ejection rate (PER) max delta between ED and ES timepoint
        diff_lv = volumes['LV'][:-1] - volumes['LV'][1:]
        lv_per_rt, lv_per_tp = np.max(diff_lv), np.argmax(diff_lv)
        if np.argmax(es_ed_indicator_vec['LV']['ES']) < lv_per_tp:
            print("Warning - {}: LV - PER - tp is greater than ES {} > {} (rt={:.2f}) ".format(p_id, lv_per_tp,
                                                                                               np.argmax(es_ed_indicator_vec['LV']['ES']),
                                                                                               lv_per_rt))
        # peak_rates[]
        diff_rv = volumes['RV'][:-1] - volumes['RV'][1:]
        rv_per_rt, rv_per_tp = np.max(diff_rv), np.argmax(diff_rv)
        if np.argmax(es_ed_indicator_vec['RV']['ES']) < rv_per_tp:
            print("Warning - {}: RV - PER - tp is greater than ES {} > {} (rt={:.2f})".format(p_id, rv_per_tp,
                                                                                              np.argmax(es_ed_indicator_vec['RV']['ES']),
                                                                                              rv_per_rt))
        peak_rates[p_id]['LV'] = {'PER': lv_per_rt}
        peak_rates[p_id]['RV'] = {'PER': rv_per_rt}
        peak_rates[p_id]['LV']['PFR'], peak_rates[p_id]['RV']['PFR'] = peak_filling_rate(volumes, es_ed_indicator_vec, p_id=p_id)

    return peak_rates


def create_excel_with_params(auto_volumes, peak_rates=None, cols=None, output_dir='~/expers/ARVC/', do_save=False,
                             filename=None):
    if cols is None and peak_rates is not None:
        cols = ['StudyID', 'LV-EF', 'LV-EDV', 'LV-ESV', 'LV-SV', 'LV-PER', 'LV-PFR',
                'RV-EF', 'RV-EDV', 'RV-ESV', 'RV-SV', 'RV-PER', 'RV-PFR']
    elif cols is None and peak_rates is None:
        print("Column header without filling rates")
        cols = ['StudyID', 'LV-EF', 'LV-EDV', 'LV-ESV', 'LV-SV',
                'RV-EF', 'RV-EDV', 'RV-ESV', 'RV-SV']

    vol_params = []
    for patid, phase_dict in auto_volumes.items():
        new_row = dict()
        for struc, params in phase_dict.items():
            if peak_rates is not None:
                p_rates = peak_rates[patid][struc]
                new_row[struc] = [params['EF'], params['EDV'], params['ESV'], params['SV'], p_rates['PER'], p_rates['PFR']]
            else:
                new_row[struc] = [params['EF'], params['EDV'], params['ESV'], params['SV']]
        table_row = [patid] + new_row['LV'] + new_row['RV']
        vol_params.append(table_row)
    vol_params = pd.DataFrame(vol_params, columns=cols)
    if do_save and output_dir is not None:
        output_dir = os.path.expanduser(output_dir)
        if filename is None:
            filename = "cmr_parameters.xlsx"
        out_file = os.path.join(output_dir, filename)
        vol_params.to_excel(out_file, index=False)
        print("INFO - Saved Excel to {}".format(out_file))
    return vol_params


def correct_slices(new_auto3d, reference3d, min_slice, max_slice, is_acdc_lbls=False):
    def convert_labels(ref_slice):
        new_ref = np.zeros_like(ref_slice).astype(ref_slice.dtype)
        for lbl_id in np.unique(ref_slice):
            if lbl_id == 0:
                continue
            indices = ref_slice == lbl_id
            new_ref[indices] = ACDC_TRANSLATE[lbl_id]
        return new_ref

    # assuming new_auto and reference are 3D np arrays
    num_slices = reference3d.shape[0]
    for slice_id in range(max_slice, num_slices):
        if not is_acdc_lbls:
            new_auto3d[slice_id] = reference3d[slice_id]
        else:
            new_auto3d[slice_id] = convert_labels(reference3d[slice_id])
    for slice_id in range(min_slice, -1, -1):
        if not is_acdc_lbls:
            new_auto3d[slice_id] = reference3d[slice_id]
        else:
            new_auto3d[slice_id] = convert_labels(reference3d[slice_id])
    return new_auto3d


def exchange_base_apex(auto_masks, ref_masks, cycle_info, filter_patid=None, is_acdc_lbls=False,
                       output_dir=None, do_save=False):
    """
        assuming both are dicts with pat id as key
    """
    new_auto_masks = defaultdict(dict)
    for patid, auto_dict in tqdm_notebook(auto_masks.items(), desc="Exchange base/apex slices", total=len(auto_masks)):
        if filter_patid is not None:
            if patid != filter_patid:
                continue
        r_mask = ref_masks[patid]['labels']
        info_ed, info_es = cycle_info[patid]['ED'], cycle_info[patid]['ES']
        new_auto_masks[patid] = copy.deepcopy(auto_masks[patid])
        n_auto = new_auto_masks[patid]['labels']
        # print("Before ", patid, np.count_nonzero(n_auto == 1), np.count_nonzero(n_auto == 2))
        if n_auto.shape != r_mask.shape:
            print("Problems - ", patid, r_mask, n_auto)
        info = {**info_ed, **info_es}
        for tp, label_ids in info.items():
            min_slice_id, max_slice_id = None, None
            # y = np.bincount(n_auto[tp].ravel())
            # ii = np.nonzero(y)[0]
            # print("Before ", tp, np.vstack((ii, y[ii])).T)
            for arvc_lbl_id in label_ids:
                ref_slice_idx = np.where(np.count_nonzero(r_mask[tp] == arvc_lbl_id, axis=(1, 2)) != 0)
                min_s_id, max_s_id = np.min(ref_slice_idx).astype(np.int), np.max(ref_slice_idx).astype(np.int)
                if min_slice_id is None or min_s_id > min_slice_id:
                    min_slice_id = min_s_id
                if max_slice_id is None or max_s_id < max_slice_id:
                    max_slice_id = max_s_id
            # IMPORTANT NOTE: because we often have 2 basal slices we correct the penultimate and last basal slice
            max_slice_id -= 1
            n_auto[tp] = correct_slices(n_auto[tp], r_mask[tp], min_slice_id, max_slice_id, is_acdc_lbls=is_acdc_lbls)
            # y = np.bincount(n_auto[tp].ravel())
            # ii = np.nonzero(y)[0]
            # print("After ", tp, np.vstack((ii, y[ii])).T)
        new_auto_masks[patid]['labels'] = n_auto

    return new_auto_masks


def get_patient_ids_ARVC(split_file="~/repo/seg_uncertainty/datasets/ARVC/train_test_split_seg.yaml") -> dict:
    """

    """
    split_file = os.path.expanduser(split_file)
    print("INFO - get_patient_ids_segmentation - Get split file "
          "from {}".format(split_file))
    # load existing splits
    with open(split_file, 'r') as fp:
        split_config = yaml.load(fp, Loader=yaml.FullLoader)
        training_ids = split_config['training']
        validation_ids = split_config['validation']
        test_ids = split_config['test']

    return {'training': training_ids, 'validation': validation_ids, 'test': test_ids}


def convert_to_bland_altman_arrays(patient_volumes):
    """
        patient_volumes (e.g. auto_volumes or ref_volumes has first key patient_id,
        second key LV / RV, third key [EF, EDV, ESV]
    """
    bland_arrays = {'LV': {'EDV': [], 'ESV': [], 'EF': [], 'SV': []},
                    'RV': {'EDV': [], 'ESV': [], 'EF': [], 'SV': []}
                    }
    patids = list(patient_volumes.keys())
    patids.sort()
    for patid in patids:
        struc_dict = patient_volumes[patid]
        for c_struc, c_indices in struc_dict.items():
            for c_ind, measure in c_indices.items():
                bland_arrays[c_struc][c_ind].append(measure)

    return bland_arrays