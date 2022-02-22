import os
import numpy as np


class Config(object):

    def __init__(self):
        self.dataset = None
        self.img_file_ext = ".nii.gz"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/'))
        self.short_axis_dir = os.path.join(self.data_root_dir, 'images')
        self.ref_label_dir = os.path.join(self.data_root_dir, 'ref_tissue_labels')
        self.tissue_structure_labels = None
        self.split_file = None
        self.datasets = ['training', 'validation', 'test']
        self.limited_load_max = 5
        self.dt_margins = None


class ConfigPIEExamples(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = ".nii.gz"
        self.data_root_dir = os.path.join(os.path.expanduser('~/data'), "cardiac_pie")
        self.short_axis_dir = os.path.join(self.data_root_dir, 'nifti/sax')
        self.ref_label_dir = None
        self.tissue_structure_labels = None
        self.limited_load_max = 5
        self.voxel_spacing_resample = np.array([1.4, 1.4]).astype(np.float32)


class ConfigARVC(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = ".nii.gz"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/' + dataset))
        self.short_axis_dir = os.path.join(self.data_root_dir, 'nifti/sax/')
        self.ref_label_dir = os.path.join(self.data_root_dir, 'annotations/contour_ref/')
        self.tissue_structure_labels = {0: 'BG', 1: 'LV', 2: 'RV'}
        self.split_file_segmentation = \
            os.path.expanduser("~/repo/seg_uncertainty/datasets/ARVC/train_test_split_seg.yaml")
        self.datasets = ['training', 'validation', 'test']
        self.limited_load_max = 3
        # transfer learning. Assuming we train on ACDC and test on ARVC. We need to translate ARVC classes
        # to ACDC classes
        self.cls_translate = {1: 3, 2: 1}


class ConfigACDC(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = ".nii.gz"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/' + dataset))
        self.short_axis_dir = os.path.join(self.data_root_dir, 'all_cardiac_phases')
        self.ref_label_dir = None
        self.split_file = os.path.join(self.data_root_dir, "train_val_test_split_sr.yaml")
        self.tissue_structure_labels = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}
        self.limited_load_max = 5
        self.voxel_spacing_resample = np.array([1.4, 1.4]).astype(np.float32)
        self.dt_margins = (4.6, 3.1)


class ConfigACDCTESTSR(Config):
    """
        This is a 4D ACDC dataset. Generated using our SR method. We synthesize every 2nd slice in the original
        volumes. Purpose, purely for segmentation evaluation as done in Xia et al.
        We compare auto segmentations on SR-resolved volumes for ED/ES with auto segmentations on original valumes
        (also ED/ES only)
        IMPORTANT:
            in-plane matrix 224x224
            volumes are already rescaled to [0, 1]. Don't rescale!!!
    """
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        # every 2nd slice is synthesized. Remaining slices are originals from 4D volumes
        self.img_file_ext = "_ni01.nii.gz"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/ACDC/'))
        self.short_axis_dir = os.path.join(self.data_root_dir, 'sr_test')
        self.ref_label_dir = None
        self.info_path = os.path.expanduser("~/data/ACDC/all_cardiac_phases/")
        self.tissue_structure_labels = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}
        self.limited_load_max = 5
        self.voxel_spacing_resample = np.array([1.4, 1.4]).astype(np.float32)


class ConfigACDCCentered(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = ".nii.gz"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/' + "ACDC"))
        self.short_axis_dir = os.path.join(self.data_root_dir, 'centered')
        self.ref_label_dir = None
        self.tissue_structure_labels = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}
        self.limited_load_max = 5
        self.voxel_spacing_resample = np.array([1.4, 1.4]).astype(np.float32)
        self.dt_margins = (4.6, 3.1)


class ConfigOASIS(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = 't88_gfc.nii.gz'
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/' + dataset))
        self.image_dir = os.path.join(self.data_root_dir, 'nifti')
        self.ref_label_dir = None
        self.tissue_structure_labels = None
        self.limited_load_max = 5


class ConfigdHCP(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = "t2w.nii.gz"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/' + dataset + "_cropped_256"))
        self.image_dir = self.data_root_dir
        self.ref_label_dir = None
        self.tissue_structure_labels = None
        self.limited_load_max = 5


class ConfigADNI(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = "_1mm.nii"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/' + dataset))
        self.image_dir = self.data_root_dir
        self.ref_label_dir = None
        self.tissue_structure_labels = None
        self.limited_load_max = 5
        self.split_file = os.path.join(self.data_root_dir, "patient_ids.yaml")


class ConfigMNIST3D(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = ".nii.gz"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/' + dataset))
        self.image_dir = self.data_root_dir
        self.ref_label_dir = None
        self.tissue_structure_labels = None
        self.limited_load_max = 100


class ConfigBrainMASI(Config):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_file_ext = ".nii"
        self.data_root_dir = os.path.abspath(os.path.expanduser('~/data/' + dataset))
        self.short_axis_dir = os.path.join(self.data_root_dir, 'BrainMASI_LR_co')
        self.ref_label_dir = None
        self.tissue_structure_labels = None
        self.limited_load_max = 5
        self.voxel_spacing_resample = np.array([1., 1.]).astype(np.float32)


def get_config(dataset="ARVC"):
    if dataset == "ARVC":
        return ConfigARVC(dataset=dataset)
    elif dataset == "ACDC" or dataset == "ACDC_full":
        if dataset == "ACDC_full":
            dataset = 'ACDC'
        return ConfigACDC(dataset=dataset)
    elif dataset == "ACDCC":
        if dataset == "ACDCC":
            dataset = 'ACDCC'
        return ConfigACDCCentered(dataset=dataset)
    elif dataset == "ACDCLBL":
        return ConfigACDC(dataset="ACDC")
    elif dataset == "ACDC4DLBL":
        return ConfigACDC(dataset="ACDC")
    elif dataset == "BrainMASI":
        return ConfigBrainMASI(dataset)
    elif dataset == "OASIS":
        return ConfigOASIS(dataset)
    elif dataset == 'dHCP':
        return ConfigdHCP(dataset)
    elif dataset == 'ADNI':
        return ConfigADNI(dataset)
    elif dataset == 'MNIST3D':
        return ConfigMNIST3D(dataset)
    elif dataset == "ACDCTESTSR":
        return ConfigACDCTESTSR(dataset=dataset)
    elif dataset == "PIE":
        return ConfigPIEExamples(dataset=dataset)
    else:
        ValueError("Error - get_config - Unknown dataset {}".format(dataset))



