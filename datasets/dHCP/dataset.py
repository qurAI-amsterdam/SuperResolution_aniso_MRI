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
from datasets.dHCP.create_dataset import get_patient_ids


def create_lr_dataset_dHCP(downsample_steps, limited_load=False):
    dataset_name = 'dHCP'
    data_config = get_config(dataset_name)
    file_suffix = data_config.img_file_ext
    file_suffix_new = get_file_suffix_blurred(dataset_name, file_suffix, downsample_steps)
    filepath_generator = Path(data_config.image_dir).rglob('*' + file_suffix)
    patid_list = [int(file_obj.name.replace(file_suffix, "").split("_")[0]) for file_obj in filepath_generator]
    # patid_list = patid_list[300:]
    images_hr = get_images(patid_list, dataset_name, limited_load=limited_load, rescale_int=False,
                           do_downsample=False, downsample_steps=downsample_steps, hr_only=True)

    for p_id, data_dict in images_hr.items():
        img_path = data_dict['img_path']
        img_lr = simulate_thick_slices(data_dict['image'], downsample_steps / 2)
        img_lr = sitk.GetImageFromArray(img_lr)
        img_lr.SetSpacing(data_dict['spacing'][::-1])
        img_lr.SetDirection(data_dict['direction'])
        img_lr.SetOrigin(data_dict['origin'])
        dirname, filename = str(Path(img_path).parent.absolute()), str(Path(img_path).name)
        fname = os.path.join(dirname, filename.replace(file_suffix, file_suffix_new))
        print("new file ", fname)
        sitk.WriteImage(img_lr, fname, True)


def get_dataset_braindHCP(args, src_path, rs=np.random.RandomState(1234), type_of_set="both",
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
        training_set = BrainDHCP('training',
                                           root_dir=src_path,
                                           rescale=True, resample=False,
                                           transform=transform_tr,
                                           limited_load=args['limited_load'],
                                           slice_selection=args['slice_selection'],
                                           downsample=downsample, downsample_steps=downsample_steps)
    if type_of_set in ['both', 'validation']:
        val_set = BrainDHCP('validation', root_dir=src_path,
                                      rescale=True, resample=False,
                                      transform=transform_te,
                                      limited_load=test_limited_load,
                            slice_selection=args['slice_selection'],
                            downsample=downsample, downsample_steps=downsample_steps)
    return training_set, val_set


class BrainDHCP(BrainDataset):

    def __init__(self, dataset,  # Training Test
                 images=None,
                 root_dir='~/data/dHCP_cropped_256/',
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
        self.dataset_name = 'dHCP'
        data_config = get_config(self.dataset_name)
        if images is None:
            patid_list = get_patient_ids(dataset, data_config.image_dir)
            print("WARNING - BrainDHCP dataset {} - downsample {} "
                  "with factor {} ({})".format(dataset, downsample, downsample_steps, self.slice_selection))

            self.images = get_images(patid_list, 'dHCP', limited_load=limited_load, rescale_int=rescale,
                                     do_downsample=downsample, downsample_steps=downsample_steps,
                                     include_hr_images=False, verbose=True)
        else:
            self.images = images
        self._get_indices()


def rename_dHCP_files(src_path="~/data/dHCP_cropped_256", orig_file_suffix="t2w.nii.gz",
                      new_file_suffix="_t2w.nii.gz"):
    src_path = os.path.expanduser(src_path)
    filepath_generator = Path(src_path).rglob('*' + orig_file_suffix)

    for fname in filepath_generator:
        s_dir, s_file = str(fname.parent.absolute()),  str(fname.name)
        # patid_dir = s_file.replace(orig_file_suffix, "")
        new_file = s_file.replace(orig_file_suffix, new_file_suffix)
        # patid_dir = os.path.join(s_dir, patid_dir)
        # if not os.path.isdir(patid_dir):
        #    os.makedirs(patid_dir, exist_ok=False)
        new_file = os.path.join(s_dir, new_file)
        print(str(fname))
        print(new_file)
        os.rename(str(fname), new_file)
