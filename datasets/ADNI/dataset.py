import numpy as np
from torchvision import transforms
from datasets.common_brains import BrainDataset, get_file_suffix_blurred, simulate_thick_slices
import os
from datasets.brainMASI.custom_transforms import RandomCropNextToCenter, GenericToTensor
from datasets.brainMASI.custom_transforms import CenterCrop
from datasets.common_brains import get_images
from datasets.data_config import get_config
import SimpleITK as sitk
from pathlib import Path
from datasets.ADNI.create_dataset import get_patient_ids


def get_dataset_brainADNI(args, src_path, rs=np.random.RandomState(1234), type_of_set="both",
                          transform_tr=None, transform_te=None, test_limited_load=True, downsample=True,
                          downsample_steps=3):
    src_path = os.path.expanduser(src_path)
    training_set = None
    val_set = None
    if transform_tr is None:
        transform_tr = transforms.Compose([RandomCropNextToCenter(args['width'], rs=rs, max_translation=35),
                                           # RandomRotation(rs=rs),
                                           GenericToTensor()])
    if transform_te is None:
        transform_te = transforms.Compose([CenterCrop(args['width']), GenericToTensor()])
    if type_of_set in ['both', 'train']:
        training_set = BrainADNI('training',
                                           root_dir=src_path,
                                           rescale=True, resample=False,
                                           transform=transform_tr,
                                           limited_load=args['limited_load'],
                                           slice_selection=args['slice_selection'],
                                           downsample=downsample, downsample_steps=downsample_steps)
    if type_of_set in ['both', 'validation']:
        val_set = BrainADNI('validation', root_dir=src_path,
                                      rescale=True, resample=False,
                                      transform=transform_te,
                                      limited_load=test_limited_load,
                            slice_selection=args['slice_selection'],
                            downsample=downsample, downsample_steps=downsample_steps)
    return training_set, val_set


class BrainADNI(BrainDataset):

    def __init__(self, dataset,  # Training Test
                 images=None,
                 root_dir='~/data/ADNI/',
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
        self.dataset_name = 'ADNI'
        data_config = get_config(self.dataset_name)
        if images is None:
            patid_list = get_patient_ids(dataset, data_config.image_dir)
            print("WARNING - {} dataset {} - downsample {} "
                  "with factor {} ({})".format(self.dataset_name, dataset, downsample, downsample_steps, self.slice_selection))

            self.images = get_images(patid_list, self.dataset_name, limited_load=limited_load, rescale_int=rescale,
                                     do_downsample=downsample, downsample_steps=downsample_steps,
                                     include_hr_images=False, verbose=True)
        else:
            self.images = images
        self._get_indices()