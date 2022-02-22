import numpy as np
import yaml
import os
from datasets.data_config import get_config
from datasets.common_brains import get_images, simulate_thick_slices
import SimpleITK as sitk
from pathlib import Path


def get_patient_ids(type_of_set, src_path="~/data/ADNI/"):
    src_path = os.path.expanduser(src_path)
    split_file = os.path.join(src_path, "patient_ids.yaml")
    if os.path.isfile(split_file):
        # load existing splits
        with open(split_file, 'r') as fp:
            patient_ids = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        print("Error - training/validation/test split file does not exist: {}".format(split_file))
        quit(1)
    if type_of_set == 'full':
        complete_list = []
        for pat_list in patient_ids.values():
            complete_list.extend(pat_list)
        return complete_list
    else:
        return patient_ids[type_of_set]


def create_lr_dataset(downsample_steps, limited_load=False, source_dir="~/data/ADNI/",
                      rename_original=False):
    dataset = 'ADNI'
    dta_config = get_config(dataset)
    file_suffix = ".nii"  # this is the original suffix
    file_suffix_new = "_{}mm.nii".format(downsample_steps)
    patid_list = get_patient_ids("full", src_path=source_dir)
    images_hr = get_images(patid_list, dataset, limited_load=limited_load, rescale_int=False,
                           do_downsample=False, downsample_steps=downsample_steps, hr_only=True,
                           src_path=source_dir)

    for p_id, data_dict in images_hr.items():
        img_path = data_dict['img_path']
        img_lr = simulate_thick_slices(data_dict['image'], downsample_steps)
        img_lr = sitk.GetImageFromArray(img_lr)
        img_lr.SetSpacing(data_dict['spacing'][::-1])
        img_lr.SetDirection(data_dict['direction'])
        img_lr.SetOrigin(data_dict['origin'])
        dirname, filename = str(Path(img_path).parent.absolute()), str(Path(img_path).name)
        original_fname = os.path.join(dirname, filename)
        fname = os.path.join(dirname, filename.replace(file_suffix, file_suffix_new))
        sitk.WriteImage(img_lr, fname, True)
        print("INFO - saved image to {}".format(fname))
        if rename_original:
            new_file = original_fname.replace('.nii', "_1mm.nii")
            print("New filename {}".format(new_file))
            os.rename(original_fname, new_file)


def rename_files(src_path="~/data/ADNI", orig_file_suffix=".nii",
                      new_file_suffix="_1mm.nii"):
    src_path = os.path.expanduser(src_path)
    filepath_generator = Path(src_path).rglob('*' + orig_file_suffix)
    i = 0
    for fname in filepath_generator:
        s_dir, s_file = str(fname.parent.absolute()),  str(fname.name)
        if '5mm' in s_file or '1mm' in s_file:
            continue
        i += 1
        # patid_dir = s_file.replace(orig_file_suffix, "")
        new_file = s_file.replace(orig_file_suffix, new_file_suffix)
        # patid_dir = os.path.join(s_dir, patid_dir)
        # if not os.path.isdir(patid_dir):
        #    os.makedirs(patid_dir, exist_ok=False)
        new_file = os.path.join(s_dir, new_file)
        print(str(fname))
        print(new_file)
        # os.rename(str(fname), new_file)
    print("INFO - Renamed {} files".format(i))