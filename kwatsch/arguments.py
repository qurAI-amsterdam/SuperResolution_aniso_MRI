import argparse
import os

OUTPUT_DIRS_SR = {'ACDC': {'ae': '~/expers/sr/ACDC/ae',
                           'alpha': '~/expers/sr/ACDC/alpha',
                           'acai': '~/expers/sr/ACDC/acai'},
                  }


def parse_args():
    parser = argparse.ArgumentParser(description='Train auto-encoder for SR')
    parser.add_argument('-d', '--dataset', type=str, choices=['ACDC', 'ACDCC', 'dHCP', 'ADNI',
                                                              'OASIS', 'MNIST3D', 'MNISTRoto',
                                                              'ACDCLBL'], default='ACDC',
                        help='Dataset to train on')
    # downsample_steps is ONLY APPLICABLE for HR dataset brain e.g. dHCP neonatal set
    parser.add_argument('--downsample_steps', type=int, default=None)
    parser.add_argument('-ss', '--slice_selection', type=str, choices=['adjacent_plus', 'adjacent', 'mix'],
                        default='adjacent_plus',  help='Way to select slices for a batch')
    parser.add_argument('-c', '--comment', type=str, default=None)
    parser.add_argument('-m', '--model', type=str, choices=['ae', 'ae_combined', 'aesr', 'aesr_combined',
                                                            'vae', 'vae_combined', 'acai', 'acai_combined',
                                                             'vae2'],
                                                   default='ae', help='Model to train')
    parser.add_argument('-id', '--exper_id', type=str, default='debug',
                        help='Determine subdir where output is stored')

    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('--model_filename', type=str, default=None)
    parser.add_argument('-e', '--epochs', type=int, default=250)
    parser.add_argument('-l', '--lr', type=float, default=0.00001)
    parser.add_argument('-w', '--weight_decay', type=float, default=0.)
    parser.add_argument('-b', '--batch_size', type=int, default=12)
    parser.add_argument('-bt', '--test_batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--limited_load', action='store_true')
    parser.add_argument('-s', '--seed', type=int, default=892372)
    parser.add_argument('-g', '--gpu_ids', type=int, nargs='+', default=[0])
    parser.add_argument('-p', '--port', type=int, default=8030)

    parser.add_argument('--number_of_workers', type=int, default=2)
    parser.add_argument('--validate_every', type=int, default=500)

    parser.add_argument('--alpha_loss_func', type=str, default=None, choices=[None, 'mse', 'perceptual'])
    parser.add_argument('--use_percept_loss', action='store_true')
    parser.add_argument('--use_ssim_loss', action='store_true')
    parser.add_argument('--use_extra_latent_loss', action='store_true')
    parser.add_argument('--use_loss_annealing', action='store_true')
    parser.add_argument('--alpha_class', type=str, default=None)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--latent_width', type=int, default=16)
    parser.add_argument('--latent', type=int, default=16)
    parser.add_argument('--depth', type=int, default=32)
    parser.add_argument('--ae_class', type=str, default="VanillaACAI")
    parser.add_argument('--image_mix_loss_func', type=str, default=None)
    parser.add_argument('--ex_loss_weight1', type=float, default=0.001)
    parser.add_argument('--lamb_reg_acai', type=float, default=0.5)  # only for ACAI model (lambda in paper)
    parser.add_argument('--vae_beta', type=float, default=None)  # only for beta-VAE model
    parser.add_argument('--aug_patch_size', type=int, default=None)
    parser.add_argument('--get_masks', action='store_true')
    parser.add_argument('--log_tensorboard', action='store_true')  # write log to tensorboard (images etc)
    parser.add_argument('--epoch_threshold', type=int, default=100, help="save models > epoch_threshold")

    args = parser.parse_args()
    if args.model == "ae_combined" and args.image_mix_loss_func is None:
        args.image_mix_loss_func = "perceptual"
        print("!!! Warning !!! - arguments - Using perceptual loss for image mix distance")
    if args.model == 'vae' or args.model == 'vae_combined':
        args.ae_class = 'VAE'
        if args.model == 'vae' and args.vae_beta is None:
            args.vae_beta, args.lamb = 100, 1.
        elif args.model == 'vae_combined ' and args.vae_beta is None:
            args.vae_beta, args.lamb = 200, 1.
        else:
            args.lamb = 1  # do nothing, vae_beta is set on command line
    elif args.model == 'vae2':
        args.ae_class = 'VAE2'
        args.lamb = 1
        if args.vae_beta is None:
            args.vae_beta = 1
    else:
        args.vae_beta, args.lamb = 0, 0
    if args.downsample_steps is None:
        raise ValueError("Error - arguments - downsample_steps cannot be None")
    if args.dataset == "OASIS" and args.aug_patch_size is None and args.width < 220:
        args.aug_patch_size = 220
        print("!!! WARNING !!! - arguments - setting aug_patch_size to {}".format(args.aug_patch_size))
    if args.dataset == "dHCP" and args.aug_patch_size is None and args.width < 256:
        args.aug_patch_size = 256
        print("!!! WARNING !!! - arguments - setting aug_patch_size to {}".format(args.aug_patch_size))
    if args.dataset in ["ACDC", 'ACDCLBL'] and args.aug_patch_size is None:
        args.aug_patch_size = 180
        print("!!! WARNING !!! - arguments - setting aug_patch_size to {}".format(args.aug_patch_size))
    if args.output_dir is not None:
        args.output_dir = os.path.expanduser(os.path.join(args.output_dir, args.exper_id))
    else:
        temp_dir = os.path.join(os.path.join(os.path.join('~/expers/sr_redo', args.dataset), args.model), args.exper_id)
        args.output_dir = os.path.expanduser(temp_dir)
    if args.model_filename is not None:
        args.model_filename = os.path.expanduser(args.model_filename)
    args_dict = vars(args)

    return args, args_dict
