import os
from tqdm import tqdm_notebook, tqdm
from evaluate.metrics import compute_ssim_for_batch, compute_psnr_for_batch, compute_lpips_for_batch
from evaluate.metrics import compute_vif_for_batch
from kwatsch.common import isnotebook
import numpy as np
from evaluate.common import determine_last_slice


def generate_synth_slices_mask(orig_num_slices, downsample_steps):
    num_slices = determine_last_slice(orig_num_slices, downsample_steps) + 1
    s_mask = np.ones(num_slices).astype(np.bool)
    r_mask = np.zeros(num_slices).astype(np.bool)
    s_mask[np.arange(0, num_slices)[::downsample_steps]] = False
    r_mask[np.arange(0, num_slices)[::downsample_steps]] = True
    # mask indicating slices that we reconstruct and s_mask indicating slices that we synthesize
    return r_mask, s_mask


def compare_quantitative(data_generator_ref, data_generator_method, interpol_method, is_4d=True,
                         downsample_steps=None, percept_criterion=None, do_save=False,
                         method_dir=None, transform=None, eval_axis=0, frame_id=None, eval_dataset=None):
    if do_save and method_dir is None:
        raise ValueError("ERROR - you need to specify output_dir for saving")
    is_conventional = True if interpol_method in ['linear', 'bspline', 'lanczos'] else False
    print("INFO - evaluating {} images from {}".format(interpol_method, method_dir))
    print("INFO - evaluation axis-{} - method: {} - is conventional {}"
          " - downsample steps {}".format(eval_axis, interpol_method, is_conventional, downsample_steps))
    ssim_res, psnr_res, lpips_res, vif_res, mse_res = [], [], [], [], []
    ssim_synth, psnr_synth, lpips_synth, vif_synth, mse_synth = [], [], [], [], []
    ssim_recon, psnr_recon, lpips_recon, vif_recon, mse_recon = [], [], [], [], []
    if isnotebook():
        loop_generator = tqdm_notebook(data_generator_ref.items(), desc="Compute metrics", total=len(data_generator_ref))
    else:
        loop_generator = tqdm(data_generator_ref.items(), desc="Compute metrics", total=len(data_generator_ref))
    for p_id, data_dict_org in loop_generator:
        if is_4d:
            if frame_id is None:
                frame_loop = np.arange(data_dict_org['image'].shape[0])
            else:
                frame_loop = np.arange(frame_id, frame_id + 1)
        else:
            frame_loop = np.arange(0, 1)
        for f_id in frame_loop:
            # 'image' key contains simluated (blurred) thick slices. We need HR data for evaluation:
            image_key = 'image_hr' if 'image_hr' in data_dict_org.keys() else 'image'
            images_ref = data_dict_org[image_key] if not is_4d else data_dict_org[image_key][f_id]
            # IMPORTANT!!! Add 1 to last_slice_id because we want to compute metrics t/m this slice_id
            if eval_axis == 0:
                last_slice_id = determine_last_slice(images_ref.shape[0], downsample_steps) + 1
            else:
                last_slice_id = images_ref.shape[0]
            # mask for reconstructed (r_mask) and synthesized slices (s_mask)
            r_mask, s_mask = generate_synth_slices_mask(images_ref.shape[0], downsample_steps=downsample_steps)
            images_synth = data_generator_method[p_id]['image'] if not is_4d else data_generator_method[p_id]['image'][f_id]
            if transform is not None:
                images_synth = transform({'image': images_synth})['image']
                images_ref = transform({'image': images_ref})['image']
            # All slices
            ssim_res.append(
                compute_ssim_for_batch(images_ref[:last_slice_id], images_synth[:last_slice_id],
                                       eval_axis=eval_axis))
            if percept_criterion is not None:
                lpips_res.append(
                    compute_lpips_for_batch(images_ref[:last_slice_id], images_synth[:last_slice_id],
                                            eval_axis=eval_axis,
                                            criterion=percept_criterion))
            psnr_res.append(
                compute_psnr_for_batch(images_ref[:last_slice_id], images_synth[:last_slice_id],
                                       eval_axis=eval_axis))
            vif_res.append(
                compute_vif_for_batch(images_ref[:last_slice_id], images_synth[:last_slice_id],
                                      eval_axis=eval_axis))
            mse_res.append(np.mean((images_ref[:last_slice_id] - images_synth[:last_slice_id])**2))

            if eval_axis == 0:
                # We collect metrics for synthesized slices only and reconstructed slices only
                # Synthesized slices only
                ssim_synth.append(
                    compute_ssim_for_batch(images_ref[:last_slice_id][s_mask], images_synth[:last_slice_id][s_mask],
                                           eval_axis=eval_axis))
                if percept_criterion is not None:
                    lpips_synth.append(
                        compute_lpips_for_batch(images_ref[:last_slice_id][s_mask], images_synth[:last_slice_id][s_mask],
                                                eval_axis=eval_axis,
                                                criterion=percept_criterion))
                psnr_synth.append(
                    compute_psnr_for_batch(images_ref[:last_slice_id][s_mask], images_synth[:last_slice_id][s_mask],
                                           eval_axis=eval_axis))
                # print("PatID {} {:.03f}".format(p_id, psnr_synth[-1]))
                vif_synth.append(
                    compute_vif_for_batch(images_ref[:last_slice_id][s_mask], images_synth[:last_slice_id][s_mask],
                                          eval_axis=eval_axis))
                mse_synth.append(np.mean((images_ref[:last_slice_id][s_mask] - images_synth[:last_slice_id][s_mask]) ** 2))
                # reconstructions only
                ssim_recon.append(
                    compute_ssim_for_batch(images_ref[:last_slice_id][r_mask], images_synth[:last_slice_id][r_mask],
                                           eval_axis=eval_axis))
                if percept_criterion is not None:
                    lpips_recon.append(
                        compute_lpips_for_batch(images_ref[:last_slice_id][r_mask],
                                                images_synth[:last_slice_id][r_mask],
                                                eval_axis=eval_axis,
                                                criterion=percept_criterion))
                psnr_recon.append(
                    compute_psnr_for_batch(images_ref[:last_slice_id][r_mask], images_synth[:last_slice_id][r_mask],
                                           eval_axis=eval_axis))
                vif_recon.append(
                    compute_vif_for_batch(images_ref[:last_slice_id][r_mask], images_synth[:last_slice_id][r_mask],
                                          eval_axis=eval_axis))
                mse_recon.append(np.mean((images_ref[:last_slice_id][r_mask] - images_synth[:last_slice_id][r_mask]) ** 2))
    mean_ssim, std_ssim = np.mean(np.array(ssim_res)), np.std(np.array(ssim_res))
    mean_psnr, std_psnr = np.mean(np.array(psnr_res)), np.std(np.array(psnr_res))
    if percept_criterion is not None:
        mean_lpips, std_lpips = np.mean(np.array(lpips_res)), np.std(np.array(lpips_res))
    else:
        mean_lpips, std_lpips = 0, 0
    mean_vif, std_vif = np.mean(np.array(vif_res)), np.std(np.array(vif_res))
    # for synthesized only slices
    if eval_axis == 0:
        mean_ssim_synth, std_ssim_synth = np.mean(np.array(ssim_synth)), np.std(np.array(ssim_synth))
        mean_psnr_synth, std_psnr_synth = np.mean(np.array(psnr_synth)), np.std(np.array(psnr_synth))
        if percept_criterion is not None:
            mean_lpips_synth, std_lpips_synth = np.mean(np.array(lpips_synth)), np.std(np.array(lpips_synth))
        else:
            mean_lpips_synth, std_lpips_synth = 0, 0
        mean_vif_synth, std_vif_synth = np.mean(np.array(vif_synth)), np.std(np.array(vif_synth))
        # reconstructed
        mean_ssim_recon, std_ssim_recon = np.mean(np.array(ssim_recon)), np.std(np.array(ssim_recon))
        mean_psnr_recon, std_psnr_recon = np.mean(np.array(psnr_recon)), np.std(np.array(psnr_recon))
        if percept_criterion is not None:
            mean_lpips_recon, std_lpips_recon = np.mean(np.array(lpips_recon)), np.std(np.array(lpips_recon))
        else:
            mean_lpips_recon, std_lpips_recon = 0, 0
        mean_vif_recon, std_vif_recon = np.mean(np.array(vif_recon)), np.std(np.array(vif_recon))
    print("{}: SSIM / PSRN / LPIPS / VIF: {:.3f} ({:.2f}) / "
          "{:.2f} ({:.2f}) / {:.3f} ({:.2f}) / {:.3f} ({:.2f})".format(interpol_method, mean_ssim, std_ssim,
                                                                       mean_psnr, std_psnr, mean_lpips, std_lpips,
                                                                       mean_vif, std_vif))
    if eval_axis == 0:
        print("{} (recon): SSIM / PSRN / LPIPS / VIF: {:.3f} ({:.2f}) / "
              "{:.2f} ({:.2f}) / {:.3f} ({:.2f}) / {:.3f} ({:.2f})".format(interpol_method, mean_ssim_recon,
                                                                           std_ssim_recon,
                                                                           mean_psnr_recon, std_psnr_recon,
                                                                           mean_lpips_recon, std_lpips_recon,
                                                                           mean_vif_recon, std_vif_recon))
        print("{} (synth): SSIM / PSRN / LPIPS / VIF: {:.3f} ({:.2f}) / "
              "{:.2f} ({:.2f}) / {:.3f} ({:.2f}) / {:.3f} ({:.2f})".format(interpol_method, mean_ssim_synth, std_ssim_synth,
                                                                           mean_psnr_synth, std_psnr_synth, mean_lpips_synth, std_lpips_synth,
                                                                           mean_vif_synth, std_vif_synth))
        save_metrics = {'ssim': np.array(ssim_res), 'psnr': np.array(psnr_res), 'vif': np.array(vif_res),
                        'lpips': np.array(lpips_res), 'mse': np.array(mse_res),
                        'ssim_res': np.array([mean_ssim, std_ssim]),
                        'psnr_res': np.array([mean_psnr, std_psnr]), 'vif_res': np.array([mean_vif, std_vif]),
                        'lpips_res': np.array([mean_lpips, std_lpips]),
                        # same for synthesized only results
                        'ssim_synth': np.array(ssim_synth), 'psnr_synth': np.array(psnr_synth),
                        'vif_synth': np.array(vif_synth),
                        'lpips_synth': np.array(lpips_synth), 'mse_synth': np.array(mse_synth),
                        'ssim_res_synth': np.array([mean_ssim_synth, std_ssim_synth]),
                        'psnr_res_synth': np.array([mean_psnr_synth, std_psnr_synth]),
                        'vif_res_synth': np.array([mean_vif_synth, std_vif_synth]),
                        'lpips_res_synth': np.array([mean_lpips_synth, std_lpips_synth]),
                        # reconstructed
                        'ssim_recon': np.array(ssim_recon), 'psnr_recon': np.array(psnr_recon),
                        'vif_recon': np.array(vif_recon), 'mse_recon': np.array(mse_recon),
                        'lpips_recon': np.array(lpips_recon),
                        'ssim_res_recon': np.array([mean_ssim_recon, std_ssim_recon]),
                        'psnr_res_recon': np.array([mean_psnr_recon, std_psnr_recon]),
                        'vif_res_recon': np.array([mean_vif_recon, std_vif_recon]),
                        'lpips_res_recon': np.array([mean_lpips_recon, std_lpips_recon]),
                        }
    else:
        save_metrics = {'ssim': np.array(ssim_res), 'psnr': np.array(psnr_res), 'vif': np.array(vif_res),
                        'lpips': np.array(lpips_res), 'ssim_res': np.array([mean_ssim, std_ssim]),
                        'psnr_res': np.array([mean_psnr, std_psnr]), 'vif_res': np.array([mean_vif, std_vif]),
                        'lpips_res': np.array([mean_lpips, std_lpips]) }

    if do_save:
        method_dir = os.path.join(method_dir, "results")
        if not os.path.isdir(method_dir):
            os.makedirs(method_dir, exist_ok=False)
        if eval_axis == 0:
            file_name = "{}_{}x.npz".format(interpol_method, downsample_steps)
        else:
            file_name = "{}_{}x_axis{}.npz".format(interpol_method, downsample_steps, eval_axis)
        if eval_dataset is not None:
            file_name = "{}_".format(eval_dataset) + file_name
        abs_file_name = os.path.join(method_dir, file_name)

        np.savez(abs_file_name, **save_metrics)
        print("INFO - Saved results to {}".format(abs_file_name))
    return save_metrics


def load_results(method_path_dict, downsample_steps, eval_axis=0, eval_dataset=None):

    meth_result_dict = {}
    for meth, m_path in method_path_dict.items():
        if eval_axis == 0:
            file_name = "{}_{}x.npz".format(meth, downsample_steps)
        else:
            file_name = "{}_{}x_axis{}.npz".format(meth, downsample_steps, eval_axis)
        if eval_dataset is not None:
            file_name = "{}_".format(eval_dataset) + file_name
        fname = os.path.join(m_path, "results" + os.sep + file_name)
        try:
            meth_result_dict[meth] = {metric:  np.load(fname)[metric] for metric in np.load(fname)}
            print("INFO - loading from {}".format(fname))
            print(meth_result_dict[meth].keys())
            print("{}: ssim {:.3f}, psnr {:.3f}, vif {:.3f}".format(meth, meth_result_dict[meth]['ssim_res'][0],
                                                                    meth_result_dict[meth]['psnr_res'][0],
                                                                    meth_result_dict[meth]['vif_res'][0]))
            if eval_axis == 0:
                print("{}: ssim-synth {:.3f}, psnr-synth {:.3f}, vif-synth {:.3f}".format(meth, meth_result_dict[meth]['ssim_res_synth'][0],
                                                                        meth_result_dict[meth]['psnr_res_synth'][0],
                                                                        meth_result_dict[meth]['vif_res_synth'][0]))
        except FileNotFoundError:
            print("Warning - no results found for {}: {}".format(meth, fname))
        try:
            combined = "ae_caisr"
            file_name = "{}_{}x_axis{}.npz".format(combined, downsample_steps, eval_axis)
            if eval_dataset is not None:
                file_name = "{}_".format(eval_dataset) + file_namecompare_quantitative
            fname = os.path.join(m_path, "results" + os.sep + file_name)
            meth_result_dict[combined] = {metric: np.load(fname)[metric] for metric in np.load(fname)}

            print(meth_result_dict[combined].keys())
            print("{}: ssim {:.3f}, psnr {:.3f}, vif {:.3f}".format(combined, meth_result_dict[combined]['ssim_res'][0],
                                                                    meth_result_dict[combined]['psnr_res'][0],
                                                                    meth_result_dict[combined]['vif_res'][0]))
        except:
            pass
    return meth_result_dict


def format_latex_string(result_dict):
   pstr = r'\begin{{tabular}} {{@{{}}c@{{}}}}{:.3f} \\ $\pm${:.2f} \end{{tabular}} &'.format(result_dict['ssim_res_recon'][0],
                                                                                     result_dict['ssim_res_recon'][1])
   pstr = pstr + r' \begin{{tabular}} {{@{{}}c@{{}}}}{:.3f} \\ $\pm${:.2f} \end{{tabular}} &'.format(result_dict['ssim_res_synth'][0],
                                                                                     result_dict['ssim_res_synth'][1])

   pstr = pstr + r' \begin{{tabular}} {{@{{}}c@{{}}}}{:.2f} \\ $\pm${:.2f} \end{{tabular}} &'.format(
       result_dict['psnr_res_recon'][0],
       result_dict['psnr_res_recon'][1])
   pstr = pstr + r' \begin{{tabular}} {{@{{}}c@{{}}}}{:.2f} \\ $\pm${:.2f} \end{{tabular}} &'.format(
       result_dict['psnr_res_synth'][0],
       result_dict['psnr_res_synth'][1])

   pstr = pstr + r' \begin{{tabular}} {{@{{}}c@{{}}}}{:.3f} \\ $\pm${:.2f} \end{{tabular}} &'.format(
       result_dict['vif_res_recon'][0],
       result_dict['vif_res_recon'][1])
   pstr = pstr + r' \begin{{tabular}} {{@{{}}c@{{}}}}{:.3f} \\ $\pm${:.2f} \end{{tabular}} \\'.format(
       result_dict['vif_res_synth'][0],
       result_dict['vif_res_synth'][1])
   print(pstr)


def load_conventional(root_dir, downsample_steps, ):
    result_dict = {}
    methods = ['linear', 'bspline', 'lanczos']
    root_dir = os.path.expanduser(root_dir)
    for m in methods:
        m_dir = os.path.join(root_dir, 'conventional' + os.sep + m + os.sep + 'results')
        f = os.path.join(m_dir, '{}_{}x.npz'.format(m, downsample_steps))
        if os.path.isfile(f):
            result_dict[m] = np.load(f)
        else:
            print("WARNING - {} not found!".format(f))
    return result_dict
