import os
import numpy as np
from torchvision import transforms
from evaluate.metrics import compute_ssim_for_batch, compute_psnr_for_batch
from evaluate.metrics import compute_vif_for_batch
from datasets.shared_transforms import CenterCrop, AdjustToPatchSize
from datasets.dHCP.create_dataset import get_images as get_images_dHCP
from datasets.brainMASI.dataset import get_images as get_images_MASI
from kwatsch.get_trainer import get_trainer


MODEL_SPECS = {
        """
                >>>>>> 1.5mm dHCP neonatal MRIs <<<<<<<<
        """
        # ***********************************  BASELINE AUTOENCODER
        "braindHCP_p256_l16_w32_1.5mm_ae":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_w32_1.5mm_ae", "156.networks")),
        # AE but this time with mix slice selection (on geer)
        "braindHCP_p256_l16_lw32_1.5mm_ae_seed2":
           tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_1.5mm_ae_seed2", "199.networks")),
        # ***********************************  ACAI
        # first seed (SIGMOID True, advdepth=16, LPIPS=[0.4, 0.3, 0.2, 0.05, 0.05])
        # 0.943 (0.01) & *34.38 (0.69) & \textbf{0.862} (0.02)
        "braindHCP_p256_l16_w32_1.5mm":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_w32_1.5mm", "213.networks")), # 481
        # second seed (SIGMOID FALSE, advdepth=32, lr-decay=220eps, LPIPS=1/1/1/1/1)
        # higher SSIM/lower VIF: 0.973 (0.00) / 34.69 (1.05) / 0.843 (0.01)
        "braindHCP_p256_l16_lw32_1.5mm_seed2":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_1.5mm_seed2", "85.networks")),  #
        # without perceptual loss (with mix slice selection)
        "braindHCP_p256_l16_lw32_1.5mm_mselap":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_1.5mm_mselap", "184.networks")),
        # ACAI with 2 reg losses from aesr model
        "braindHCP_p256_l16_lw32_1.5mm_exreg":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_1.5mm_exreg", "194.networks")),  # 200
        # *********************************** Constraint AESR
        # first seed: (SIGMOID False, lr-decay=None, LPIPS=[0.4, 0.3, 0.2, 0.05, 0.05])
        "braindHCP_p256_l16_lw32_1.5mm_aesr":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_1.5mm_aesr", "200.networks")), # 200
        # second seed: (SIGMOID False, lr-decay=None, LPIPS=[0.4, 0.3, 0.2, 0.05, 0.05])
        "braindHCP_p256_l16_lw32_1.5mm_aesr_seed2":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_1.5mm_aesr_seed2", "159.networks")),
        "braindHCP_p256_l16_lw32_1.5mm_aesr_seed2_ft":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_1.5mm_aesr_seed2_ft", "84.networks")),

        "braindHCP_p256_l16_lw32_1.5mm_aesr_mselap":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_1.5mm_aesr_mselap", "214.networks")),
        "braindHCP_p256_l16_lw32_1.5mm_aesr_exloss":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_1.5mm_aesr_exloss", "280.networks")),


        """
                3mm dHCP neonatal MRIs        
        """
        "braindHCP_p32_l16_w32_3mm":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p32_l16_w32_3mm/", "3885.networks")),
         "braindHCP_p64_l16_w32_3mm":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p64_l16_w32_3mm", "1415.networks")),
        "braindHCP_p128_l16_w32_3mm":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p128_l16_w32_3mm", "240.networks")),
        "braindHCP_p256_l16_w32_3mm":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_w32_3mm", "451.networks")),
        # Re-run 3mm acai with different seed. Should be same setting as braindHCP_p256_l16_w32_3mm
        # GEER: 200, 446,
        "braindHCP_p256_l16_w32_3mm_seed2":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_3mm_seed2", "200.networks")),
"braindHCP_p256_l16_lw32_3mm_aesr":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_3mm_aesr", "230.networks")),  # 214
        # fine tuned AESR (model 240)
        "braindHCP_p256_l16_lw32_3mm_aesr_ft":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_3mm_aesr_ft", "15.networks")),
        "braindHCP_p256_l16_lw32_3mm_aesr_exloss":
        tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_lw32_3mm_aesr_exloss", "206.networks")),
"braindHCP_p256_l16_w32_3mm_ae":
       tuple(("~/expers/brain_sr/brain_dHCP256_LR_ax/p256_l16_w32_3mm_ae", "301.networks")),
        """
            
        """
        "brainMASI_p256_l16_w32_f3":
        tuple(("~/expers/brain_sr/brain_MASI256_ax/p256_l16_w32_f3", "446.networks")),
}


def evaluate_on_interpol_images(interpol_method, EVAL_ID, file_suffix=None, type_of_set="Test", eval_dataset="braindHCP",
                   eval_axis=0, pat_id=None, eval_patch_size=256, out_dir=None, resol=None):

    assert interpol_method is None or EVAL_ID is None
    if eval_dataset == "braindHCP":
        src_data_path = os.path.expanduser("~/data/dHCP_cropped_256")
        data_generator = get_images_dHCP(type_of_set, src_path=src_data_path, patid=pat_id,
                                         int_perc=tuple((0, 100)), rescale_int=True)
    else:
        src_data_path = os.path.expanduser("~/data/BrainMASI_cropped")
        data_generator = get_images_MASI(type_of_set, src_path=src_data_path, patid=pat_id,
                                         int_perc=tuple((0, 100)), rescale_int=True)
    if interpol_method not in ['linear', 'bspline', 'lanczos', 'nearest']:
        exper_specs = MODEL_SPECS[EVAL_ID]
        _, myargs = get_trainer(exper_specs, args_only=True)
        print("WARNING - Current model >>> {} <<< "
              "train patch-size/test patch-size {}/{}".format(EVAL_ID, myargs['width'],
                                                              eval_patch_size))
        src_path_method = os.path.join(myargs['output_directory'], out_dir)
    else:
        src_path_method = os.path.join(src_data_path + "_recons" + resol, out_dir)
    transform = transforms.Compose([AdjustToPatchSize((eval_patch_size, eval_patch_size)),
                                    CenterCrop(eval_patch_size)])
    print("INFO - Comparing original HR dataset against volumes in {}".format(src_path_method))
    print("WARNING - eval patch size = {}".format(eval_patch_size))
    data_generator_recon = get_images_dHCP(None, src_path=src_path_method, patid=pat_id,
                                           int_perc=tuple((0, 100)), rescale_int=True,
                                           file_suffix=file_suffix)
    ssim_res, psnr_res, vif_res = [], [], []
    print("INFO - #patients to process {}".format(len(data_generator.values())))
    for data_dict_hr in data_generator.values():
        patid = data_dict_hr['patient_id']
        data_dict_hr = transform(data_dict_hr)
        data_dict = data_generator_recon[patid]
        data_dict = transform(data_dict)
        ssim_res.append(compute_ssim_for_batch(data_dict_hr['image'], data_dict['image'],
                                               eval_axis=eval_axis, normalize=False))
        psnr = compute_psnr_for_batch(data_dict_hr['image'], data_dict['image'],
                                      eval_axis=eval_axis, normalize=False)
        vif = compute_vif_for_batch(data_dict_hr['image'], data_dict['image'],
                                    eval_axis=eval_axis, normalize=False)
        if not (np.isnan(vif) or np.isinf(vif)):
            vif_res.append(vif)
        if not (np.isnan(psnr) or np.isinf(psnr)):
            psnr_res.append(psnr)
        else:
            print("WARNING PSNR not a number ", psnr)
    return ssim_res, psnr_res, vif_res
