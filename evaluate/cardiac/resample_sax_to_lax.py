import torch
import numpy as np
from torch import inverse as tr_inv
from kwatsch.create_nifti_from_dicom import get_2d_from3d, get_3d_from4d
import SimpleITK as sitk


def resample_sax_to_lax(sax_image: sitk.Image, target_shape: tuple,
                        transformed_ident_grid: torch.FloatTensor):
    assert len(target_shape) == 4
    num_frames_sax = sax_image.GetSize()[::-1][0]
    res_img4d = np.zeros(target_shape).astype(np.float32)
    for frame_id in np.arange(num_frames_sax).astype(np.int):
        sax_img_3d = get_3d_from4d(sax_image, int(frame_id))
        sax_3d = sitk.GetArrayFromImage(sax_img_3d)
        sax_3d_torch = torch.from_numpy(sax_3d.astype(np.float32))
        resampled_sax = torch.nn.functional.grid_sample(sax_3d_torch[None, None],
                                                        transformed_ident_grid[None],
                                                        align_corners=True).cpu().detach().squeeze().numpy()
        res_img4d[frame_id] = resampled_sax

    return res_img4d


def make_transform(ident_grid, lax_shape, sax_shape,
                   tr_S_lax, tr_R_lax, tr_T_lax,
                   tr_S_sax,  tr_R_sax, tr_T_sax):
    world_grid = ident_grid.reshape(lax_shape[0], -1, 4) @ tr_S_lax.T @ tr_R_lax.T @ tr_T_lax.T
    # we need to bring coordinates to
    sax_coord = world_grid @ tr_inv(tr_T_sax).T @ tr_inv(tr_R_sax).T @ tr_inv(tr_S_sax).T
    # it should NOT be necessaary to scale because sax coordinates are without spacing
    # after inverse(S_sax)
    # sax_coord = sax_coord / np.array(sax_spacing)[None, None, None].astype(np.float32)
    scaled_sax_coord = (sax_coord / \
                        ((np.r_[sax_shape[::-1], 2][None, None, None].astype(np.float32) - 1) / 2)) - 1
    scaled_sax_coord = scaled_sax_coord.reshape(tuple(lax_shape) + (4,))
    # get rid off homogenous coords
    scaled_sax_coord = scaled_sax_coord[..., :3]
    return scaled_sax_coord


def make_identity_grid(shape, stackdim='last', device='cpu'):
    # Assuming shape: [z, y, x]
    if isinstance(stackdim, int):
        dim = stackdim
    elif stackdim == 'last':
        dim = len(shape)
    elif stackdim == 'first':
        dim = 0
    else:
        Exception('Incorrect stackdim given.')

    coords = [torch.arange(0, s, dtype=torch.float32, device=device) for s in shape]

    grids = torch.meshgrid(coords)
    # we need to reverse ordering of last dim for coordinates because shape is
    # z, y, x and we want (x, y, z) coords
    return torch.stack(grids[::-1], dim=dim)


def make_lax_identity_grid(target_shape):
    # target_shape is [z, y, x] but target_spacing [x, y, z]
    grid = make_identity_grid(target_shape)
    # for homogeneous coordinates we need to add fourth dim with constand value "1"
    h_coords = torch.ones(grid.shape[:-1] + (1,))
    grid = torch.cat([grid, h_coords], dim=-1)
    return grid