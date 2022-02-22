import os
from torchvision import transforms
from evaluate import compute_ssim_for_batch, compute_psnr_for_batch, compute_lpips_for_batch
from evaluate import compute_vif_for_batch
from datasets.shared_transforms import CenterCrop, AdjustToPatchSize
from datasets.ACDC.data import acdc_all_image4d
from datasets.common import get_images_in_dir
from kwatsch.get_trainer import get_trainer


MODEL_SPECS = {
            'acdc_p128_l16_16_12':
             tuple(("~/expers/cardiac_sr/acdc_new_simple/p128_l16_16_12/", "561.networks")),

              'acdc_p128_l16_16_pp_batchn':
             tuple(("~/expers/cardiac_sr/acdc_new_simple/p128_l16_16_pp_batchn/", "247.networks")),

            'acdc_p192_l16_16_thick_async_augs':
         tuple(("~/expers/cardiac_sr/acdc_new_simple/p192_l16_16_thick_async_augs", "117.networks")),

            'acdc_p128_l16_32_thick_trainp128':
            tuple(("~/expers/cardiac_sr/acdc_new_simple/p128_l16_32_batchnn_thick", "191.networks")),
            'acdc_p32_l16_32_batchnn_thick':
            tuple(("~/expers/cardiac_sr/acdc256/p32_l16_32_batchnn_thick", "526.networks")),
            'acdc_p32_l16_32_batchnn_thick_sigmoid':
            tuple(("~/expers/cardiac_sr/acdc256/p32_l16_32_batchnn_thick_sig", "167.networks")),
            'acdc_p64_l16_32_batchnn_thick':
            tuple(("~/expers/cardiac_sr/acdc256/p64_l16_32_batchnn_thick", "166.networks")),

            # again p128 but trained on images with max width/height 140x140
            'acdc128_p128_l16_32_batchnn':
            tuple(("~/expers/cardiac_sr/acdc128/p128_l16_32_batchnn", "171.networks")),


        # trained with spectral norm layers
        "acdc_p128_lw16_l16_w32_acai_mselap":
        tuple(("~/expers/cardiac_sr/acdc128/p128_lw16_l16_w32_acai_mselap", "100.networks")),  # 139
        'acdc_centered_p128_lw16_l16_w32':
        tuple(("~/expers/cardiac_sr/acdc_c128/p128_lw16_l16_w32", "154.networks")),  # 154  134
          'arvc_p128_l16_32_batchnn_sig':
        tuple(("~/expers/cardiac_sr/arvc128/p128_l16_32_batchnn_sig", "149.networks")),
        # IMPORTANT: only these two use batchnorm in discriminator
        "acdc_p128_l16_32_disc32batchnn":
        tuple(("~/expers/cardiac_sr/acdc128/p128_l16_32_disc32batchnn", "26.networks")),
    # Constraint AE (version 1)
    'acdc_p128_lw16_l16_w32_aesr':
    tuple(("~/expers/cardiac_sr/acdc128/p128_lw16_l16_w32_aesr", "150.networks")),   # 161
    # acdc_p128_lw16_l16_w32_aesr fine-tuned with extra loss
    'acdc_p128_lw16_l16_w32_aesr_ft':
    tuple(("~/expers/cardiac_sr/acdc128/p128_lw16_l16_w32_aesr_ft", "66.networks")),
    # trained on mix of adjacent and "in-between" slices. Extra loss on in-between slices
    'acdc_p128_lw16_l16_w32_aesr_exloss':
     tuple(("~/expers/cardiac_sr/acdc128/p128_lw16_l16_w32_aesr_exloss", "174.networks")), # 105
    'acdc_p128_lw16_l16_w32_aesr_exloss_ft':
     tuple(("~/expers/cardiac_sr/acdc128/p128_lw16_l16_w32_aesr_exloss_ft", "175.networks")),
    # standard AE without adversarial regularizer
    'acdc_p128_l16_32_ae':
    tuple(("~/expers/cardiac_sr/acdc128/p128_l16_32_ae", "121.networks")),
    # second seed for AE
    "acdc_p128_lw16_l16_w32_ae_seed2":
    tuple(("~/expers/cardiac_sr/acdc128/p128_lw16_l16_w32_ae_seed2", "122.networks")),
    'acdc_p128_l16_32_ae_mse_lap':
    tuple(("~/expers/cardiac_sr/acdc128/p128_l16_32_ae_mse_lap", "124.networks")),
    'acdc_p128_l16_32_ae_mse':
    tuple(("~/expers/cardiac_sr/acdc128/p128_l16_32_ae_mse", "124.networks")),
    # MORE COMPRESSION
    'acdc_p128_lw10_l20_w32_rfs':
    tuple(("~/expers/cardiac_sr/acdc128/p128_lw10_l20_w32_rfs", "102.networks")),
    'acdc_p128_lw12_l20_w32':
    tuple(("~/expers/cardiac_sr/acdc128/p128_lw12_l20_w32", "160.networks")),

    # ******************************** ACAI
    # 0.839 (0.040) / 26.46 (1.958) / 0.859 (0.023)
    'acdc_p128_l16_32_batchnn_thick':
        tuple(("~/expers/cardiac_sr/acdc128/p128_l16_32_batchnn_thick", "199.networks")),
    # ACAI seed 2: 0.837 (0.04) / 26.17 (1.96) / 0.851 (0.02) (model 110)
    #              0.836 (0.04) / 26.36 (1.95) / 0.851 (0.02) (model 132)
    "acdc_p128_lw16_l16_w32_seed2":
    tuple(("~/expers/cardiac_sr/acdc128/p128_lw16_l16_w32_seed2", "132.networks"))
}

from lpips.perceptual import PerceptualLoss


def evaluate_on_interpol_images(interpol_method, EVAL_ID, file_suffix=None, eval_dataset="ACDC",
                                pat_list=None, eval_patch_size=128, out_dir=None, compute_lpips=False):

    assert interpol_method is None or EVAL_ID is None
    downsample_steps = None
    conv_interpol = False
    ROOT_DIR = '~/data/ACDC/all_cardiac_phases' if eval_dataset == 'ACDC' else '~/data/ACDC/centered'
    transform = transforms.Compose([AdjustToPatchSize((eval_patch_size, eval_patch_size)),
                                    CenterCrop(eval_patch_size)])
    if compute_lpips:
        criterion = PerceptualLoss(
                model='net-lin', net='vgg', use_gpu='cuda', gpu_ids=[0])

    if interpol_method is None and EVAL_ID is not None:
        exper_specs = MODEL_SPECS[EVAL_ID]
        _, myargs = get_trainer(exper_specs, args_only=True)
        print("WARNING - Current model >>> {} <<< "
              "train patch-size/test patch-size {}/{}".format(EVAL_ID, myargs['width'],
                                                              eval_patch_size))
        src_path_method = os.path.join(myargs['output_directory'], out_dir)
    else:
        src_path_method = os.path.join(
            os.path.expanduser("~/data/{}{}_recons5mm".format(eval_dataset, eval_patch_size)), out_dir)
    data_generator = acdc_all_image4d(root_dir=os.path.expanduser(ROOT_DIR), resample=True,
                                      rescale=True, new_spacing=tuple((1, 1.4, 1.4)),
                                      limited_load=False, patid_list=pat_list)
    print("INFO - Comparing original 5mm volumes against {}".format(src_path_method))
    data_generator_recons = get_images_in_dir(src_path_method, dataset_name="ACDC", file_suffix=file_suffix,
                                              rescale_int=True,
                                              do_downsample=False,
                                              downsample_steps=None,
                                              patid_list=pat_list)
    ssim_res, psnr_res, vif_res, lpips_res = [], [], [], []
    for data_dict_hr in data_generator:
        patid = data_dict_hr['patient_id']
        frame_id = data_dict_hr['frame_id']
        images_hr = data_dict_hr['image']
        images_hr_recon = data_generator_recons[patid]['image'][frame_id]
        images_hr = transform({'image': images_hr})['image']
        images_hr_recon = transform({'image': images_hr_recon})['image']
        ssim_res.append(compute_ssim_for_batch(images_hr, images_hr_recon, eval_axis=0,
                                               downsample_steps=downsample_steps,
                                               conv_interpol=conv_interpol))
        vif_res.append(compute_vif_for_batch(images_hr, images_hr_recon, eval_axis=0,
                                             downsample_steps=downsample_steps,
                                             conv_interpol=conv_interpol))
        psnr_res.append(compute_psnr_for_batch(images_hr, images_hr_recon, eval_axis=0,
                                               downsample_steps=downsample_steps,
                                               conv_interpol=conv_interpol))
        if compute_lpips:
            lpips_res.append(compute_lpips_for_batch(images_hr, images_hr_recon, criterion=criterion))

    return ssim_res, psnr_res, vif_res, lpips_res