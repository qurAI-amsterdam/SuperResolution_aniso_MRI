import gpustat
import numpy as np
import torch
from torchvision.utils import make_grid
from torch import autograd
from imageio import imsave


def save_image_grid(img_grid, filename):
    if img_grid.shape[2] != 3:
        img_grid = img_grid.transpose(1, 2, 0)
    img_grid = np.uint8(np.clip(img_grid, 0, 255))
    imsave(filename, img_grid)


def generate_batch_compare_grid(batch_dict, s_mix_inbetween, recon_images):
    s_inbetween = batch_dict['slice_between']
    img1, img3 = batch_dict['image'].split(batch_dict['image'].size(0) // 2, dim=0)
    recon1, recon3 = recon_images.split(recon_images.size(0) // 2, dim=0)
    diff = s_inbetween - s_mix_inbetween
    img_grid = torch.cat([img1, recon1, s_inbetween,
                          s_mix_inbetween, diff, recon3, img3], dim=0)
    img_grid = make_grid(img_grid.detach().cpu(), s_inbetween.size(0), padding=2,
                         normalize=False, pad_value=0.5).numpy()
    return img_grid


def showMemoryUsage(device=1):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print('Used/total: ' + "{}/{}".format(item["memory.used"], item["memory.total"]))


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, dim, gp_lambda, device='cuda'):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement( ) /batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, dim, dim)
    alpha = alpha.to(device)

    fake_data = fake_data.view(batch_size, 3, dim, dim)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty
