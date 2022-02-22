import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import matplotlib as mpl
from torchvision.utils import make_grid
from imageio import imsave


def make_interp_image_grid(interp_images, num_interpolations, normalize=True):
    b = interp_images.size(0)
    num_rows = num_interpolations + 2
    assert b % num_rows == 0
    width = b // num_rows
    return make_grid(interp_images, width, num_rows, normalize=normalize).numpy(), width, num_rows


def latent_space_interp_diff_patch_size(alpha, trainer, img1, img2, patch_size, device='cuda') -> torch.FloatTensor:

    def tile_image(l_image, patch_size, new_batch_size):
        img_patches = l_image.unfold(0, *patch_size).unfold(1, *patch_size)
        return torch.reshape(img_patches, (new_batch_size, 1, patch_size[0], patch_size[1]))
    # images are pytorch tensors. If they already have batch format [z, 1, y, x] we need to squeeze the first dim
    if img1.dim() == 4:
        img1 = img1.squeeze(dim=1)
    if img2.dim() == 4:
        img2= img2.squeeze(dim=1)

    num_slices, w, h = img1.shape
    new_batch_size = w // patch_size[0] * h // patch_size[1]
    # latent vector of first image
    if not img1.is_cuda:
        img1 = img1.to(device)

    # latent vector of second image
    if not img2.is_cuda:
        img2 = img2.to(device)
    interpolated_images = None

    for s in np.arange(num_slices):
        img_patches1, img_patches2 = tile_image(img1[s], patch_size, new_batch_size), tile_image(img2[s], patch_size, new_batch_size)
        z1, z2 = trainer.encode(img_patches1, use_sr_model=True), trainer.encode(img_patches2, use_sr_model=True)
        z_interp = z1 * (1 - alpha) + z2 * alpha
        interpol_recons = trainer.decode(z_interp, use_sr_model=True)
        interpol_recons = make_grid(interpol_recons, int(np.sqrt(interpol_recons.size(0))), 0)[0]
        interpolated_images = torch.cat(
            [interpolated_images, interpol_recons[None]]) if interpolated_images is not None else interpol_recons[
            None]
    if interpolated_images.dim() == 3:
        interpolated_images = torch.unsqueeze(interpolated_images, dim=1)

    return interpolated_images.detach().cpu()


def latent_space_interp(alpha, trainer, img1, img2, device='cuda', hierarchical=False,
                        with_labels=False) -> dict:

    # latent vector of first image
    if not img1.is_cuda:
        img1 = img1.to(device)
    latent_1 = trainer.encode(img1, use_sr_model=True)

    # latent vector of second image
    if not img2.is_cuda:
        img2 = img2.to(device)
    latent_2 = trainer.encode(img2, use_sr_model=True)

    # interpolation of the two latent vectors
    if hierarchical:
        inter_latent_t = alpha * latent_1['enc_top'] + (1 - alpha) * latent_2['enc_top']
        inter_latent_b = alpha * latent_1['enc_bottom'] + (1 - alpha) * latent_2['enc_bottom']
        inter_latent = {'enc_top': inter_latent_t, 'enc_bottom': inter_latent_b}
    else:
        inter_latent = alpha * latent_1 + (1 - alpha) * latent_2

    # reconstruct interpolated image
    decode_result = trainer.decode(inter_latent, use_sr_model=True)
    if with_labels:
        inter_image = decode_result['image']
        inter_label = decode_result['pred_labels']
        inter_label = inter_label.detach().cpu()
    else:
        inter_image = decode_result
        inter_label = None
    inter_image = inter_image.detach().cpu()

    return {'inter_image': inter_image, 'inter_label': inter_label}


def visualise_interp_output(image_grid, height=5, width=5, do_save=False, do_show=True,
                            output_dir=None, epoch=None):
    # images has is torch tensor with shape [batch, channels, w, h]
    # we assume batch dim is multiple of num_rows (e.g. 18, num_rows=3)
    # Goals: we plot a grid of 6 x 3 (columns x rows) in order to visualize per column two cardiac MRI slices and
    #        one interpolated slice that is located in between both slices (sort of super-resolution)

    if do_save or do_show:
        mpl.use('Agg')
        plt.figure(figsize=(width, height))
        plt.imshow(np.transpose(image_grid, (1, 2, 0)), cmap=cm.gray)

    if do_save:
        if epoch is not None:
            fig_name = "img_interp_e{}".format(epoch)
        else:
            fig_name = "img_interp"

        fig_name = os.path.join(output_dir, fig_name + ".png")
        image_grid = (image_grid * 255).astype(np.uint8)
        imsave(fig_name, image_grid.transpose((1, 2, 0)))
    if do_show:
        plt.show()

    plt.close()


def generate_mix_coefficient_plot(image_grid, height=5, width=5, do_save=False, do_show=True,
                                    output_dir=None, epoch=None):
    # images has is torch tensor with shape [batch, channels, w, h]
    # we assume batch dim is multiple of num_rows (e.g. 18, num_rows=3)
    # Goals: we plot a grid of 6 x 3 (columns x rows) in order to visualize per column two cardiac MRI slices and
    #        one interpolated slice that is located in between both slices (sort of super-resolution)

    fig = plt.figure(figsize=(12, 48))

    if do_save or do_show:
        mpl.use('Agg')
        plt.figure(figsize=(width, height))
        plt.imshow(np.transpose(image_grid, (1, 2, 0)), cmap=cm.gray)

    if do_show:
        plt.show()

    if do_save or do_show:
        plt.close()
