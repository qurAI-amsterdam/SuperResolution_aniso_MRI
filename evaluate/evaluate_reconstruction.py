import numpy as np
import torch
from evaluate import compute_psnr_for_batch, compute_ssim_for_batch, compute_vif_for_batch
from evaluate import eval_on_different_patch_size


def evaluate_model_reconstruction(trainer, myargs, data_generator, transform=None,
                                  eval_on_bigger_patch=False, compute_vif=False, normalize=True):

    ssim_results = []
    psnr_results = []
    vif_results = []
    mean_vif = -1
    for test_batch in data_generator:
        if transform is not None:
            test_batch = transform(test_batch)
        t_images = test_batch['image']
        if eval_on_bigger_patch:
            recon_s = eval_on_different_patch_size(trainer, t_images,
                                                   patch_size=tuple((myargs['width'], myargs['width'])))
        else:
            recon_s = trainer.predict(t_images[:, None].to('cuda'))
        # Clamping the values between 0 and 1 is essential when we normalize the images for evaluation
        recon_s = torch.clamp(recon_s, min=0, max=1.)
        ssim_results.append(compute_ssim_for_batch(t_images, recon_s, normalize=normalize))
        psnr_results.append(compute_psnr_for_batch(t_images, recon_s, normalize=normalize))
        if compute_vif:
            vif_results.append(compute_vif_for_batch(t_images, recon_s, normalize=normalize))

    mean_ssim = np.mean(np.array(ssim_results))
    mean_psnr = np.mean(np.array(psnr_results))
    if compute_vif:
        mean_vif = np.mean(np.array(vif_results))
    return mean_ssim, mean_psnr, mean_vif
