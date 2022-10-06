# Autoencoding low-resolution MRI for semantically smooth interpolation of anisotropic MRI


## License

GNU General Public License v3.0+ (see LICENSE file or https://www.gnu.org/licenses/gpl-3.0.txt for full text)

Copyright 2022, Amsterdam UMC location University of Amsterdam, Biomedical Engineering and Physics, Meibergdreef 9, Amsterdam, the Netherlands


## Journal publication

[DOI](https://doi.org/10.1016/j.media.2022.102393)


## Train models

### Train model on ACDC dataset (cardiac cine MRI)

The model is trained with 4D CMRIs of 70 patients as described in journal article.

````
CUDA_VISIBLE_DEVICES=0 python train_cardiac_aesr.py --dataset=ACDC --model=ae_combined --batch_size=12 --test_batch_size=16 --latent=128 --downsample_steps=2 --epochs=900 --aug_patch_size=160 --epoch_threshold=500 --exper_id mse_perc_p32_l128_ex01 --ex_loss_weight1=0.05 --output_dir=<output_dir>
````

*Note*: Replace <output_dir> with designated output directory of the experiment.

### Neonatal brain MRI (dHCP dataset)

`
CUDA_VISIBLE_DEVICES=0 python train_aesr.py --dataset=dHCP --model=ae_combined --batch_size=8 --test_batch_size=32 --latent=128 --latent_width=64 --width=256 --exper_id=pool2_w256_l128_w001_ex01 --downsample_steps=4 --epochs=750 --ex_loss_weight1=0.001 --epoch_threshold=100
`

**Note**: For journal article we trained on neonatal brain MRIs with different synthetic anisotropy ([1, 1.5, 2, 2.5, 3]mm). The original OASIS MRIs have
isotropic resolution of 0.5mm. We generated synthetic volumes with low through-plane resolution by forehand (as decribed in journal article)
To load the correct low-resolution images, you need to specify parameter `--downsample_steps` using the values [2, 3, 4, 5, 6] for the different resolutions.

**Note**: Important parameter `ex_loss_weight1` specifies hyper-parameter lambda (see journal Eq. 3), to balance reconstruction and synthesis loss.

### Adult brain MRI (OASIS dataset)

`
CUDA_VISIBLE_DEVICES=0 python train_aesr.py --dataset=OASIS --model=ae_combined --batch_size=16 --test_batch_size=32 --latent=128 --latent_width=16 --width=64 --exper_id=mse_perc_pool2_w64_l128_4mm_w001_ex01 --downsample_steps=4 --epochs=1500 --aug_patch_size=220 --ex_loss_weight1=0.001
`

**Note**: For journal article we trained on adult brain MRIs with different synthetic anisotropy ([2, 3, 4, 5, 6]mm). The original OASIS MRIs have
isotropic resolution of 1mm. We generated synthetic volumes with low through-plane resolution by forehand (as decribed in journal article)
To load the correct low-resolution images, you need to specify parameter `--downsample_steps` using the values [2, 3, 4, 5, 6] for the different resolutions.

**Note**: Synthetic low-resolution volumes can be re-created (if needed) with function `create_lr_dataset` in `datasets.OASIS.dataset.py`.

## Find best performing model on validation set

During training model weights (and optimizer state) are saved at the end of each epoch (in output_dir/models).
To determine the best performing (saved) model on the validation set, we evaluate all saved models on the 
validation set. Run the following commands assuming that ````~/expers/sr/ACDC/ae_combined/mse_perc_p32_l128_ex01```` contains all experimental files
generated during training.

`
export PYTHONPATH=~/repo/SuperResolution_aniso_MRI/
CUDA_VISIBLE_DEVICES=1 python evaluate/find_best_model.py --exper_dir=~/expers/sr/ACDC/ae_combined/mse_perc_p32_l128_ex01 --epoch_range 500 900 --eval_patch_size=128 --eval_axis=0 --dataset ACDC
`

Evaluation computes metrics (SSIM, PSNR, VIF) for (a) reconstructed (b) synthesized images and (c) for all images (reconstructed + synthesized).
Because reconstructed images are irrelevant (we only care about the quality of the synthesized images), the model with the best
metrics on the synthesized images should be chosen. 


## Use best performing model to super-resolve anisotropic MRI volumes

1. `exper_dir`: absolute path to experiment folder (contains settings.yaml and models directory)
2. `save`: if specified save high-resolution images to `output_dir`. 
3. `data_input_dir`: directory that contains the low-resolution MRIs to process
4. `model_nbr`: integer actually specifying the epoch (the model was saved, also see previous step)

`
CUDA_VISIBLE_DEVICES=1 python generate_hr_volumes.py  --exper_dir <exper_dir> --save --output_dir <output_dir> --data_input_dir <data_input_dir> --model_nbr <epoch_nbr>
`

## Evaluate model 

### Evaluate model trained on cardiac MRI (ACDC) 

To evaluate model on ACDC test set use ipython notebook `notebooks/evaluate_cardiac.ipynb`

1. Run the two cells below heading *Generate ACDC volumes with Autoencoder approach*
2. In the second cell replace: (a) `exper_src_path` with directory of the experiment (b) `model_nbr` with the best performing epoch model (see previous step)
3. Furthermore, make sure the following parameters are set correctly:

      - `output_dir = <output_dir>` (replace)
      - `resample = True` (Autoencoder was trained with 1.4x1.4mm in-plane. Setting to True will make sure volume will be resampled to original resolution)
      - `eval_axis = 0`
      - `do_save = True`
      - `eval_patch_size = 224`
      - `generate_inbetween_slices = True`
      - `use_original_slice = False`
      - `downsample_steps = 2`

### Evaluate model trained on neonatal brain MRI (dHCP) and adult brain MRI (OASIS)

To evaluate model on dHCP and OASIS test set use ipython notebook `notebooks/evaluate_brain.ipynb`

1. Run the two cells below heading *Generate volumes with Autoencoder approach*
2. In the second cell replace: (a) `exper_src_path` with directory of the experiment (b) `model_nbr` with the best performing epoch model (see previous step)
3. Furthermore, make sure the following parameters are set correctly:

      - `dataset = 'dHCP'` (valid values ['dHCP', 'OASIS'])
      - `pat_list = None` (if set to None evaluation will use test sets of both datasets)
      - `output_dir = <output_dir>` (replace)
      - `eval_axis = 0`
      - `save_volumes = True`
      - `generate_inbetween_slices = True`
      - `use_original_slice = False`
      - `downsample_steps = <int>` (set to myargs['downsample_steps'] as specified in notebook)