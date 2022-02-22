import numpy as np
import torch
import os
from tqdm import tqdm_notebook
from lpips.perceptual import PerceptualLoss
from kwatsch.get_trainer import get_trainer
from evaluate.find_best_model import get_transforms
from datasets.ACDC.data4d_simple import ACDCDataset4DPairs
from datasets.data_config import get_config
from datasets.ACDC.data4d_simple import prepare_batch_pairs as prepare_batch_pairs_acdc
from datasets.common_brains import prepare_batch_pairs as prepare_batch_pairs_brain
from datasets.dHCP.dataset import BrainDHCP
from torchvision import transforms
from datasets.shared_transforms import GenericToTensor


def lerp(a, b, t):
    return a + (b - a) * t


def get_batch_loader(dataset, batch_size=16):

    # data_sampler = torch.kwatsch.data.RandomSampler(dataset, replacement=True, num_samples=n_sample)
    data_sampler = torch.utils.data.SequentialSampler(dataset)
    batch_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=data_sampler,
                                               drop_last=True)
    return batch_loader


def get_dHCP_dataset(src_path="~/data/dHCP_cropped_256/", type_of_set="Test", limited_load=False,
                     downsample_steps=None):
    downsample = True if downsample_steps is not None else False
    src_path = os.path.expanduser(src_path)
    transform = transforms.Compose([GenericToTensor()])
    return BrainDHCP(type_of_set, root_dir=src_path,
                        rescale=True, resample=False,
                        transform=transform,
                        limited_load=limited_load,
                        slice_selection='adjacent_plus',
                        downsample=downsample, downsample_steps=downsample_steps)


def get_dataset(type_of_set, patch_size, limited_load=False, images4d=None, slice_selection="mix"):
    dta_settings = get_config('ACDC')
    transform = get_transforms(transform_patch_size=patch_size, to_tensor=True)
    return ACDCDataset4DPairs(type_of_set,
                              fold=0,
                              images4d=images4d,
                              root_dir=dta_settings.short_axis_dir,
                              resample=True,
                              transform=transform,
                              limited_load=limited_load,
                              slice_selection=slice_selection,
                              new_spacing=tuple((1, 1.4, 1.4)),
                              thick_slices_only=True)


def compute_ppl(src_path, model_nbr, batch_loader, device='cuda', n_samples=1000, eps=1e-4):
    src_path = os.path.expanduser(src_path)
    model_nbr = str(model_nbr) + ".networks" if isinstance(model_nbr, int) else model_nbr
    trainer, exp_args = get_trainer(tuple((src_path, model_nbr)), eval_mode=True)
    if trainer.percept_criterion is None:
        trainer.percept_criterion = PerceptualLoss(
            model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
        )
    ppl(trainer, batch_loader, n_samples, eps=eps)


def normalize(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))


def ppl(trainer, batch_loader, n_samples, device='cuda', eps=1e-4):
    distances = []
    with torch.no_grad():
        i = 0
        for batch_dict in tqdm_notebook(batch_loader):
            if "num_frames_vol" in batch_dict.keys():
                batch_dict = prepare_batch_pairs_acdc(batch_dict, expand_type="repeat")
            else:
                batch_dict = prepare_batch_pairs_brain(batch_dict, expand_type="repeat")
            image = batch_dict['image'].to(device)
            image_between = batch_dict['slice_between'].to(device)
            lerp_t = torch.rand(image.size(0) // 2, device=device) // 2

            # Sanity check
            # enc = trainer.encode(image)
            # enc0, enc1 = enc.split(enc.shape[0] // 2)
            # lerp_t.fill_(0.5)
            # latent_mix = lerp(enc0, enc1, lerp_t[:, None, None, None])
            # image_mix = trainer.decode(latent_mix)
            # image_between = normalize(image_between)
            # image_mix = normalize(image_mix)
            # dist = trainer.percept_criterion(image_between, image_mix, normalize=True)
            # PPL computation
            image = trainer.encode(image)
            enc0, enc1 = image.split(image.shape[0] // 2)
            latent_e0 = lerp(enc0, enc1, lerp_t[:, None, None, None])
            latent_e1 = lerp(enc0, enc1, lerp_t[:, None, None, None] + eps)
            latent_e = torch.cat([latent_e0, latent_e1], dim=0)
            image = trainer.decode(latent_e, use_sr_model=True)
            image_a, image_b = image.split(image.shape[0] // 2)

            # Important: normalize needs to be set to True because images are already rescaled between [0, 1]
            dist = trainer.percept_criterion(image_a, image_b, normalize=True) / (
                   eps ** 2
            )
            distances.append(dist.to('cpu').numpy())
            i += 1
            if i >= n_samples:
                break

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation='lower')
    hi = np.percentile(distances, 99, interpolation='higher')
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    print('ppl: %2.3f (%2.3f) / filtered %2.3f (%2.3f)' % (distances.mean(), distances.std(),
                                                           filtered_dist.mean(), filtered_dist.std()))
    return distances