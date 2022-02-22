import torch
from torchvision.utils import make_grid

import numpy as np
import copy
from evaluate.create_HR_images import create_hr_images


def test_interpolation_with_rigid_transformations(trainer, images, num_interpol=9, transform=None):
    """

    """
    target_images = copy.deepcopy(images)
    target_images = transform({'image': target_images.detach().cpu().numpy()})['image']
    target_images = torch.from_numpy(target_images)
    if images.dim() == 3:
        # assuming x has [z, y, x] and is missing channel dimension. Insert default one
        images = images[:, None]
        target_images = target_images[:, None]
    if images.dtype != 'cuda':
        images = images.to('cuda')
        target_images = target_images.to('cuda')
    z1 = trainer.model.encode(images)
    z2 = trainer.model.encode(target_images)
    z1, z2 = z1.data.cpu().numpy(), z2.data.cpu().numpy()

    z_interp = [z1 * (1 - t) + z2 * t for t in np.linspace(0, 1, num_interpol + 2)[1:-1]]
    z_interp = np.vstack(z_interp)
    x_interp = trainer.model.decode(torch.FloatTensor(z_interp).to(images.device))
    x_interp = x_interp.cpu().data.numpy()

    all = []
    all.extend(images.data.cpu().numpy())
    all.extend(x_interp)
    all.extend(target_images.data.cpu().numpy())
    all = torch.from_numpy(np.asarray(all))

    img_grid = make_grid(all, images.shape[0], padding=2, normalize=False, pad_value=0.5).numpy().squeeze().transpose(1, 2, 0)
    return img_grid


def evaluate_interpolation_performance(trainer, myargs, data_generator, transform=None,
                                       downsample_steps=None, file_suffix=None, patient_id=None, eval_axis=0):

    compute_percept_loss = False
    save_volumes = False
    use_original_slice = False  # in create_hr_images default = False
    normalize = False
    generate_inbetween_slices = True
    num_interpolations = downsample_steps - 1
    is_4d = True if myargs['dataset'] in ["ACDC", "ARVC"] else False

    result_dict = create_hr_images(data_generator, myargs, trainer,
                                                   num_interpolations=num_interpolations,
                                                   downsample_steps=downsample_steps,
                                                   use_original_slice=use_original_slice,
                                                   is_4d=is_4d, transform=transform, normalize=normalize,
                                                   generate_inbetween_slices=generate_inbetween_slices,
                                                   patient_id=patient_id, file_suffix=file_suffix,
                                                   save_volumes=save_volumes, eval_axis=eval_axis,
                                                   compute_percept_loss=compute_percept_loss, verbose=False)

    return result_dict
