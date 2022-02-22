import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from datasets.common_brains import BrainDataset
from glob import glob
import os
import SimpleITK as sitk
from datasets.common import rescale_intensities, get_random_adjacent_slice
from datasets.shared_transforms import RandomRotation, RandomIntensity, GenericToTensor


def get_dataset_brainMASI(args, src_path, rs=np.random.RandomState(1234), type_of_set="both",
                          transform_tr=None, transform_te=None, test_limited_load=True, downsample=True,
                          downsample_steps=3):
    # IMPORTANT !!! For brainMASI we currently only have training and test set !!!
    src_path = os.path.expanduser(src_path)
    training_set = None
    val_set = None
    if transform_tr is None:
        transform_tr = transforms.Compose([RandomRotation(rs=rs), RandomIntensity(rs=rs),
                                           GenericToTensor()])

    if transform_te is None:
        transform_te = transforms.Compose([GenericToTensor()])
    if type_of_set in ['both', 'train']:
        training_set = BrainMASI('training',
                                           root_dir=src_path,
                                           rescale=True, resample=False,
                                           transform=transform_tr,
                                           limited_load=args['limited_load'],
                                           slice_selection=args['slice_selection'],
                                           downsample=downsample, downsample_steps=downsample_steps)
    if type_of_set in ['both', 'test']:
        val_set = BrainMASI('test', root_dir=src_path,
                                      rescale=True, resample=False,
                                      transform=transform_te,
                                      limited_load=test_limited_load,
                            slice_selection=args['slice_selection'],
                            downsample=downsample, downsample_steps=downsample_steps)
    return training_set, val_set


def get_images(dataset="Training", src_path="~/data/BrainMASI_LR_co/", limited_load=False, patid=None,
               rescale_int=True, int_perc=tuple((0, 100)), do_downsample=False, downsample_steps=3,
               file_suffix=".nii"
               ):
    src_path = os.path.expanduser(src_path)
    search_path = os.path.join(src_path, dataset + os.sep + "images" + os.sep + "*" + file_suffix)
    file_list = glob(search_path)
    file_list.sort()
    image_dict = {}
    if limited_load:
        file_list = file_list[:3]

    for fname in file_list:
        base_fname = os.path.basename(fname)
        patient_id = int(base_fname.replace(".nii", ""))
        if patid is not None:
            if patient_id != patid:
                continue
        img = sitk.ReadImage(fname)
        orig_spacing = np.array(img.GetSpacing()[::-1]).astype(np.float64)
        origin = np.array(img.GetOrigin()).astype(np.float64)
        direction = np.array(img.GetDirection()).astype(np.float64)
        np_img = sitk.GetArrayFromImage(img).astype(np.float64)
        if do_downsample:
            np_img = np_img[::int(downsample_steps)]
        if rescale_int:
            np_img = rescale_intensities(np_img, percs=int_perc)
        image_dict[patient_id] = {'image': np_img, 'spacing': orig_spacing, 'orig_spacing': orig_spacing,
                                      'origin': origin, 'direction': direction,
                                      'patient_id': patient_id, "num_slices": np_img.shape[0]}
    return image_dict


class BrainMASI(BrainDataset):

    def __init__(self, dataset,  # Training Test
                 images=None,
                 root_dir='~/data/BrainMASI_cropped/',
                 transform=None, limited_load=False,
                 rescale=True,
                 resample=False,
                 rs=np.random.RandomState(1234),
                 slice_selection="adjacent_plus",  # "adjacent_plus"   "mix"
                 downsample=True,
                 downsample_steps=None):
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
            print("WARNING - BrainMASI dataset {} - downsample {} with factor {}".format(dataset, downsample,
                                                                                         downsample_steps))
            self.images = get_images(dataset, root_dir, limited_load=limited_load, rescale_int=rescale,
                                     do_downsample=downsample, downsample_steps=downsample_steps)
        else:
            self.images = images
        self._get_indices()



