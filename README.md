# Autoencoding low-resolution MRI for semantically smooth interpolation of anisotropic MRI


## License

GNU General Public License v3.0+ (see LICENSE file or https://www.gnu.org/licenses/gpl-3.0.txt for full text)

Copyright 2022, Amsterdam UMC location University of Amsterdam, Biomedical Engineering and Physics, Meibergdreef 9, Amsterdam, the Netherlands


## Journal publication

[DOI](https://doi.org/10.1016/j.media.2022.102393)


## Train model on ACDC dataset (cardiac cine MRI)

The model is trained with 4D CMRIs of 70 patients as described in journal article.

````
CUDA_VISIBLE_DEVICES=1 python train_cardiac_aesr.py --dataset=ACDC --model=ae_combined --batch_size=12 --test_batch_size=16 --latent=128 --downsample_steps=2 --epochs=900 --aug_patch_size=160 --epoch_threshold=500 --exper_id mse_perc_p32_l128_ex01 --ex_loss_weight1=0.05 --output_dir=<output_dir>
````

*Note*: Replace <output_dir> with designated output directory of the experiment.


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


