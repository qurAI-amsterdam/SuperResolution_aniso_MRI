import os
import SimpleITK as sitk
import numpy as np
from datasets.common_brains import BrainDataset, get_images, get_transforms_brain, simulate_thick_slices
import pandas as pd
from pathlib import Path


def get_oasis_patient_ids(dataset, split_file_name="~/data/OASIS/oasis_cross_sectional.xlsx",
                          len_training_set=200, len_validation_set=20, len_test_set=50):
    assert dataset in ['training', 'validation', 'test', 'full']
    filename = os.path.expanduser(split_file_name)
    patient_data = pd.read_excel(filename, index_col=False)
    patient_list = patient_data['ID'].tolist()
    patient_list.sort()
    patient_list = [int(patid.replace("OAS1_", "").replace("_MR1", "").replace("_MR2", "")) for patid in patient_list]
    if dataset == "training":
        return patient_list[:len_training_set]
    elif dataset == "validation":
        return patient_list[len_training_set:len_training_set+len_validation_set]
    elif dataset == "test":
        return patient_list[-len_test_set:]
    else:
        return patient_list


def get_dataset_brainOASIS(args, src_path, rs=np.random.RandomState(1234), type_of_set="both",
                          transform_tr=None, transform_te=None, test_limited_load=True, downsample=True,
                          downsample_steps=3):
    src_path = os.path.expanduser(src_path)
    training_set = None
    val_set = None
    if transform_tr is None:
        transform_tr, transform_te = get_transforms_brain('OASIS')
    if type_of_set in ['both', 'train']:
        training_set = BrainOASIS('training',
                                           root_dir=src_path,
                                           rescale=True, resample=False,
                                           transform=transform_tr,
                                           limited_load=args['limited_load'],
                                           slice_selection=args['slice_selection'],
                                           downsample=downsample, downsample_steps=downsample_steps)
    if type_of_set in ['both', 'validation']:
        val_set = BrainOASIS('validation', root_dir=src_path,
                                      rescale=True, resample=False,
                                      transform=transform_te,
                                      limited_load=test_limited_load,
                                      slice_selection=args['slice_selection'],
                                      downsample=downsample, downsample_steps=downsample_steps)
    return training_set, val_set


class BrainOASIS(BrainDataset):
    """
        average voxel matrix of training set: [176. 208. 176.] Currently padding to 220**2 in-plane
    """

    def __init__(self, dataset,  # Training Test
                 images=None,
                 root_dir='~/data/OASIS/nifti',
                 transform=None, limited_load=False,
                 rescale=True,
                 resample=False,
                 rs=np.random.RandomState(1234),
                 slice_selection="adjacent_plus",  # "adjacent_plus"   "mix"
                 downsample=True,
                 downsample_steps=1):
        assert slice_selection in ['adjacent', 'adjacent_plus', 'mix', 'random']
        self._root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self._resample = resample
        self.dataset = dataset
        self.rs = rs
        self.slice_selection = slice_selection
        self.downsample = downsample
        self.downsample_steps = downsample_steps
        if images is None:
            file_suffix = "t88_gfc.nii.gz"
            if downsample:
                t_suffix = file_suffix.replace(".nii.gz", "")
                file_suffix = t_suffix + "_{}mm.nii.gz".format(downsample_steps)
            print("WARNING - BrainOASIS - Loading MRIs with extension {}".format(file_suffix))
            patid_list = get_oasis_patient_ids(dataset)
            print("WARNING - OASIS dataset {} ({} vols) - downsample {} "
                  "with factor {} ({})".format(dataset, len(patid_list), downsample, downsample_steps,
                                               self.slice_selection))
            self.images = get_images(patid_list, "OASIS", limited_load=limited_load, rescale_int=rescale,
                                     do_downsample=downsample, downsample_steps=downsample_steps,
                                     include_hr_images=False)
        else:
            self.images = images
        self._get_indices()

    # overwrite base method because you always use downsample steps of 2 for regularization of in-between slices
    def _get_slice_step(self):
        if self.slice_selection == "adjacent":
            return 1
        elif self.slice_selection == "adjacent_plus":
            return 2
        elif self.slice_selection == "mix":
            return self.rs.choice([1, 2])


def create_lr_dataset(downsample_steps, root_dir="~/data/OASIS/nifti", limited_load=False):
    file_suffix = "t88_gfc.nii.gz"
    t_suffix = file_suffix.replace(".nii.gz", "")
    file_suffix_new = t_suffix + "_{}mm.nii.gz".format(downsample_steps)
    patid_list = get_oasis_patient_ids("full")
    images_hr = get_images(patid_list, "OASIS", limited_load=limited_load, rescale_int=False,
                           do_downsample=False, downsample_steps=downsample_steps, hr_only=True)

    for p_id, data_dict in images_hr.items():
        img_path = data_dict['img_path']
        img_lr = simulate_thick_slices(data_dict['image'], downsample_steps)
        img_lr = sitk.GetImageFromArray(img_lr)
        img_lr.SetSpacing(data_dict['spacing'][::-1])
        img_lr.SetDirection(data_dict['direction'])
        img_lr.SetOrigin(data_dict['origin'])
        dirname, filename = str(Path(img_path).parent.absolute()), str(Path(img_path).name)
        fname = os.path.join(dirname, filename.replace(file_suffix, file_suffix_new))
        sitk.WriteImage(img_lr, fname, True)
        print("INFO - saved image to {}".format(fname))


def rename_files(src_path="~/data/OASIS/nifti", orig_file_suffix="3mm_t88_gfc.nii.gz",
                 new_file_suffix="t88_gfc_3mm.nii.gz"):
    src_path = os.path.expanduser(src_path)
    filepath_generator = Path(src_path).rglob('*' + orig_file_suffix)

    for fname in filepath_generator:
        s_dir, s_file = str(fname.parent.absolute()),  str(fname.name)
        new_file = s_file.replace(orig_file_suffix, new_file_suffix)
        new_file = os.path.join(s_dir, new_file)
        print(str(fname))
        print(new_file)
        os.rename(str(fname), new_file)