import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import torch
import numpy as np
from evaluate.vifvec import vifp_mscale
try:
    from skimage.measure import compare_ssim as ssim_metric
    from skimage.measure import compare_psnr as psnr_metric
except ImportError:
    from skimage.metrics import structural_similarity as ssim_metric
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from evaluate.common import determine_last_slice


def get_slice(img3d, np_myslice, k=0):
    img_slice = img3d[np_myslice]
    if k != 0:
        return np.rot90(img_slice, k)
    else:
        return img_slice


def determine_slice(eval_axis, slice_id):
    assert eval_axis in [0, 1, 2]
    if eval_axis == 0:
        myslice = np.s_[slice_id]
    elif eval_axis == 1:
        myslice = np.s_[:, slice_id]
    else:
        myslice = np.s_[:, :, slice_id]
    return myslice


def plot_compare(o_img_dict, syn_img_dict_m1, syn_img_dict_m2, patient_id, percept_criterion=None,
                 downsample_steps=None, frame_id=None, do_save=False, fig_dir=None, slice_range=None,
                 m1_desc=None, m2_desc=None, eval_axis=0, num_rot=0, image_hr_key=None,
                 transform=None):
    down_steps = downsample_steps
    image_hr_key = 'image' if image_hr_key is None else image_hr_key
    num_rot = 0 if eval_axis == 0 else num_rot
    if frame_id is not None:
        orig_img = o_img_dict[patient_id][image_hr_key][frame_id]
        syn_img_m1 = syn_img_dict_m1[patient_id]['image'][frame_id]
        syn_img_m2 = syn_img_dict_m2[patient_id]['image'][frame_id]
    else:
        orig_img = o_img_dict[patient_id][image_hr_key]
        syn_img_m1 = syn_img_dict_m1[patient_id]['image']
        syn_img_m2 = syn_img_dict_m2[patient_id]['image']
    if transform is not None:
        orig_img = transform({'image': orig_img})['image']
        syn_img_m1 = transform({'image': syn_img_m1})['image']
        syn_img_m2 = transform({'image': syn_img_m2})['image']
    fig_name_w_slice_range = True
    if slice_range is None:
        fig_name_w_slice_range = False
        slice_range = np.arange(0, orig_img.shape[eval_axis])
    z_spacing = o_img_dict[patient_id]['spacing'][0]
    num_ex, num_cols = len(slice_range), 3
    height = num_ex * 3
    fig, ax = plt.subplots(num_ex, num_cols, figsize=(10, height))
    last_sync_slice_id = np.arange(orig_img.shape[0])[::downsample_steps][-1]
    ssim_res_m1, ssim_res_m2, psnr_res_m1, psnr_res_m2, lpips_res_m1, lpips_res_m2 = [], [], [], [], [], []
    vif_res_m1, vif_res_m2 = [], []
    last_slice_id = determine_last_slice(orig_img.shape[0], downsample_steps)
    print("INFO - #slices/last slice {} / {} / {}".format(orig_img.shape[0], last_slice_id, last_sync_slice_id))
    for axis_idx, s_id in enumerate(slice_range):
        myslice = determine_slice(eval_axis, s_id)
        if percept_criterion is not None:
            lpips_m1 = percept_criterion(torch.from_numpy(orig_img[myslice][None, None]),
                                                        torch.from_numpy(syn_img_m1[myslice][None, None]),
                                                        normalize=True).mean().item()
            lpips_m2 = percept_criterion(torch.from_numpy(orig_img[myslice][None, None]),
                                                     torch.from_numpy(syn_img_m2[myslice][None, None]),
                                                     normalize=True).mean().item()
        else:
            lpips_m1, lpips_m2 = 0, 0
        ssim_m1 = ssim_metric(orig_img[myslice], syn_img_m1[myslice])
        psnr_m1 = psnr_metric(orig_img[myslice], syn_img_m1[myslice])
        ssim_m2 = ssim_metric(orig_img[myslice], syn_img_m2[myslice])
        psnr_m2 = psnr_metric(orig_img[myslice], syn_img_m2[myslice])
        vif_m1 = vifp_mscale(orig_img[myslice], syn_img_m1[myslice], do_rescale=True)
        vif_m2 = vifp_mscale(orig_img[myslice], syn_img_m2[myslice], do_rescale=True)
        ax[axis_idx][0].imshow(get_slice(syn_img_m1, myslice, num_rot), cmap=cm.gray, vmin=0, vmax=1,
                               interpolation="nearest")
        ax[axis_idx][1].imshow(get_slice(orig_img, myslice, num_rot), cmap=cm.gray, vmin=0, vmax=1,
                               interpolation="nearest")
        ax[axis_idx][2].imshow(get_slice(syn_img_m2, myslice, num_rot), cmap=cm.gray, vmin=0, vmax=1,
                               interpolation="nearest")
        ax[axis_idx][0].axis("off"), ax[axis_idx][1].axis("off"), ax[axis_idx][2].axis("off")
        if eval_axis == 0:
            title_m1 = m1_desc + " Reconstructed" if (s_id % down_steps == 0 or s_id >= last_sync_slice_id) else m1_desc + " Synthesized"
            title_m2 = m2_desc + " Reconstructed" if (s_id % down_steps == 0 or s_id >= last_sync_slice_id) else m2_desc + " Synthesized"
        else:
            title_m1 = "M1"
            title_m2 = "M2"
        ax[axis_idx][1].set_title("Original slice {}".format(s_id))
        ax[axis_idx][0].set_title(title_m1)
        ax[axis_idx][2].set_title(title_m2)
        # ax[axis_idx][0].set_title(title_m1 + ": {:.3f}, {:.3f}, {:.3f}".format(ssim_m1, psnr_m1,
        #                                                                    lpips_m1))
        # ax[axis_idx][2].set_title(title_m2 + ": {:.3f}, {:.3f}, {:.3f}".format(ssim_m2,
        #                                                                       psnr_m2,
        #                                                                       lpips_m2))
        if eval_axis == 0 and s_id % 2 != 0 and s_id < last_sync_slice_id or \
                eval_axis != 0:
            ssim_res_m1.append(ssim_m1), ssim_res_m2.append(ssim_m2)
            psnr_res_m1.append(psnr_m1), psnr_res_m2.append(psnr_m2)
            lpips_res_m1.append(lpips_m1), lpips_res_m2.append(lpips_m2)
            vif_res_m1.append(vif_m1), vif_res_m2.append(vif_m2)
    #
    ssim_res_m1, ssim_res_m2 = np.mean(np.array(ssim_res_m1)), np.mean(np.array(ssim_res_m2))
    psnr_res_m1, psnr_res_m2 = np.mean(np.array(psnr_res_m1)), np.mean(np.array(psnr_res_m2))
    vif_res_m1, vif_res_m2 = np.mean(np.array(vif_res_m1)), np.mean(np.array(vif_res_m2))
    if percept_criterion is not None:
        lpips_res_m1, lpips_res_m2 = np.mean(np.array(lpips_res_m1)), np.mean(np.array(lpips_res_m2))
    else:
        lpips_res_m1, lpips_res_m2 = 0, 0
    str_patid = "{}".format(patient_id) if isinstance(patient_id, str) else "{:06d}".format(patient_id)
    fig.suptitle("Patient {} ({}x) - {}/{}: SSIM {:.3f}/{:.3f}, PSNR {:.3f}/{:.3f}, "
                 " LPIPS {:.3f}/{:.3f} \n VIF {:.3f}/{:.3f} (z-spacing {:.1f}mm) "
                 "Per column SSIM/PSNR/LPIPS ".format(str_patid, downsample_steps,
                                                     'M1' if m1_desc is None else m1_desc,
                                                     'M2' if m2_desc is None else m2_desc,
                                                     ssim_res_m1, ssim_res_m2, psnr_res_m1,
                                                     psnr_res_m2, lpips_res_m1, lpips_res_m2,
                                                     vif_res_m1, vif_res_m2, z_spacing))
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save and fig_dir is not None:
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir, exist_ok=False)
        if isinstance(patient_id, int):
            patient_id = "patient{:03d}".format(patient_id)
        if frame_id is not None:
            fig_file_name = os.path.join(fig_dir, patient_id + "_f{:02d}_{}x.png".format(frame_id, downsample_steps))
        else:
            if fig_name_w_slice_range:
                s1, s2 = int(slice_range[0]), int(slice_range[-1])
                fig_file_name = os.path.join(fig_dir, patient_id + "_s{}_s{}_{}x.png".format(s1, s2, downsample_steps))
            else:
                fig_file_name = os.path.join(fig_dir, patient_id + "_{}x.png".format(downsample_steps))
        plt.savefig(fig_file_name, bbox_inches='tight')
        print(("INFO - Successfully saved fig %s" % fig_file_name))
    plt.show()





