import os
from tqdm import tqdm
from kwatsch.common import saveExperimentSettings
from shutil import copytree, rmtree
import torch
from torchvision import transforms
import numpy as np
from kwatsch.arguments import parse_args
from kwatsch.get_trainer import get_trainer, get_trainer_dynamic
from networks.net_config import NetworkConfig
from datasets.ACDC.data4d_simple import MyToTensor
from datasets.shared_transforms import RandomCrop
from datasets.shared_transforms import CenterCrop, RandomIntensity, AdjustToPatchSize
from datasets.shared_transforms import RandomRotation, GenericToTensor
from datasets.ACDC.data4d_simple import get_new_dataset_acdc, get_data_loaders_acdc
from datasets.ACDC.data4d_simple import prepare_batch_pairs
from datasets.data_config import get_config
from datasets.ARVC.dataset_sr import get_arvc_sr_dataset
from datasets.ACDC.data_with_labels import get_dataset_acdc_with_lables, get_4d_edes_image_array
from datasets.ACDC.data4d_simple import get_patids_acdc_sr, get_4d_image_array


def merge_args_architecture(args_dict, architecture):
    for key, value in architecture.items():
        # arguments passed on command line supersede config loaded from architecture (net_config.py)
        # BUT ONLY, if argments are not NONE. Boolean argments are False by default and hence, need to be passed
        # on command line to supersede architecture. This is important e.g. use_sigmoid!
        if key not in args_dict.keys() or (args_dict[key] is None):
            args_dict[key] = value
    return args_dict


def get_datasets(transform_tr, transform_te):
    global args_dict, rs
    dta_settings = get_config(args_dict['dataset'])
    if args_dict['dataset'] == "ACDC" or args_dict['dataset'] == "ACDCC":
        training_set, val_set = get_new_dataset_acdc(args_dict, dta_settings, rs, acdc_set="both",
                                                     new_spacing=tuple((1, 1.4, 1.4)),
                                                     transform_tr=transform_tr,
                                                     transform_te=transform_te,
                                                     get_masks=args_dict['get_masks'])
        pat_nums_val = get_patids_acdc_sr("validation", rs=rs, limited_load=False)
        pat_nums_val.sort()
        # choose three patients for validation
        pat_nums_val = [pat_nums_val[0], pat_nums_val[len(pat_nums_val) // 4], pat_nums_val[len(pat_nums_val) // 2],
                        pat_nums_val[-1]]
        # pat_nums_val = [26]
        # also get whole 4d volumes for showing interpolation of complete 3D volume
        image4d_val = get_4d_image_array(dta_settings.short_axis_dir, dataset=None,
                                         rescale=True, resample=True,
                                         limited_load=True, new_spacing=tuple((1, 1.4, 1.4)), rs=rs,
                                         pat_nums=pat_nums_val,
                                         get_masks=args_dict['get_masks'])
    elif args_dict['dataset'] == "ACDCLBL":
        training_set, val_set = get_dataset_acdc_with_lables(args_dict, dta_settings, rs, acdc_set="both",
                                                                new_spacing=tuple((1, 1.4, 1.4)),
                                                                transform_tr=transform_tr,
                                                                transform_te=transform_te)
        pat_nums_val = get_patids_acdc_sr("validation", rs=rs, limited_load=False)
        pat_nums_val.sort()
        # choose three patients for validation
        pat_nums_val = [pat_nums_val[0], pat_nums_val[len(pat_nums_val) // 4], pat_nums_val[len(pat_nums_val) // 2],
                        pat_nums_val[-1]]
        # also get whole 4d volumes for showing interpolation of complete 3D volume
        image4d_val = get_4d_edes_image_array(dta_settings.short_axis_dir, dataset=None,
                                         rescale=True, resample=True,
                                         limited_load=True, new_spacing=tuple((1, 1.4, 1.4)), rs=rs,
                                         pat_nums=pat_nums_val)
    elif args_dict['dataset'] == "ARVC":
        training_set, val_set = get_arvc_sr_dataset(args, rs, sets="both",
                                                    new_spacing=tuple((1, 1.25, 1.25)),
                                                    transform_tr=transform_tr,
                                                    transform_te=transform_te,
                                                    resample=True,
                                                    test_limited_load=True)
        image4d_val = None
    else:
        raise ValueError("ERROR dataset unknown {}".format(args_dict['dataset']))

    return training_set, val_set, image4d_val


def get_transforms():
    global args_dict, rs
    if args_dict['dataset'] == 'ACDCLBL':
        slice_mask = np.array([1, 0, 1, 0, 1, 0]).astype(np.bool)
        print("WARNING - !!! get_transforms !!! - Using slice mask ", slice_mask)
    else:
        slice_mask = None
    transform_tr = transforms.Compose([
        AdjustToPatchSize(tuple((args_dict['aug_patch_size'],
                                 args_dict['aug_patch_size']))),
        CenterCrop(tuple((args_dict['aug_patch_size'], args_dict['aug_patch_size']))),
        RandomCrop(args_dict['width'], rs=rs),
        RandomIntensity(rs=rs, slice_mask=slice_mask),
        RandomRotation(rs), GenericToTensor()])
        # MyToTensor()])

    if args_dict['width'] <= 128 and 'vae' not in args_dict['model']:  # vae has fixed bottleneck!
        transform_te = transforms.Compose([CenterCrop(tuple((args_dict['aug_patch_size'], args_dict['aug_patch_size']))),
                                           GenericToTensor()])  # MyToTensor()])
    else:
        transform_te = transforms.Compose([AdjustToPatchSize(tuple((args_dict['width'], args_dict['width']))),
                                           CenterCrop(args_dict['width']), GenericToTensor()])
    return transform_tr, transform_te


def generate_epoch_range():
    global args_dict
    if "model_filename" in args_dict.keys() and args_dict['model_filename'] is not None:
        state_dict = torch.load(os.path.expanduser(args_dict['model_filename']))
        last_epoch = state_dict['epoch']
        print("Warning - Retraining - fetching last epoch {}".format(last_epoch))
        return np.arange(last_epoch + 1, last_epoch + args_dict['epochs'] + 1)
    else:
        return np.arange(1, (args_dict['epochs'] + 1))


def prepare_run():
    global args_dict, model_config
    if not os.path.isdir(args_dict['output_dir']):
        os.makedirs(args_dict['output_dir'], exist_ok=False)
    elif args_dict['exper_id'] == "debug":
        print("WARNING - prepare_run - Removing output dir {}".format(args_dict['output_dir']))
        rmtree(args_dict['output_dir'])
        os.makedirs(args_dict['output_dir'], exist_ok=False)
    else:
        raise IsADirectoryError("ERROR - directory {} for experiment {} already exists. Remove first"
                                " or choose another exper_id".format(args_dict['output_dir'], args_dict['exper_id']))

    if args_dict['exper_id'] != "debug":
        copytree('./', os.path.join(args_dict['output_dir'], "src"))
    saveExperimentSettings(args, os.path.join(args_dict['output_dir'], 'settings.yaml'))
    # create dirs for images stored every now and then. Same for models
    args_dict['dir_images'] = os.path.join(args_dict['output_dir'],  'log_images')
    if not os.path.isdir(args_dict['dir_images']):
        os.makedirs(args_dict['dir_images'], exist_ok=False)
    args_dict['dir_models'] = os.path.join(args_dict['output_dir'], 'models')
    if not os.path.isdir(args_dict['dir_models']):
        os.makedirs(args_dict['dir_models'], exist_ok=False)
    print("INFO - train AE for SR - using output dir {}".format(args_dict['output_dir']))


def prepare_validation_batch(test_loader):
    global args_dict
    validation_batch = test_loader.next()
    validation_batch = prepare_batch_pairs(validation_batch, expand_type="repeat")
    validation_batch['image'] = validation_batch['image'].to(args_dict['device'])
    validation_batch['loss_mask'] = validation_batch['loss_mask'].float()
    return validation_batch


def main():
    global args, args_dict, rs
    prepare_run()
    transform_tr, transform_te = get_transforms()
    training_set, val_set, image_dict_val = get_datasets(transform_tr, transform_te)
    train_loader = get_data_loaders_acdc(training_set, args_dict, rs=rs)
    test_loader = get_data_loaders_acdc(val_set, args_dict, rs=rs, shuffle=True)
    test_loader = iter(test_loader)
    validation_batch = prepare_validation_batch(test_loader)
    trainer = get_trainer_dynamic(args_dict, model_file=args_dict['model_filename'])
    trainer.init_tensorboard(args_dict['output_dir'])
    epoch_range = generate_epoch_range()
    num_train_samples = len(train_loader.dataset)
    num_it_per_epoch = num_train_samples // args_dict['batch_size']
    show_every = args_dict['validate_every'] if num_it_per_epoch > args_dict['validate_every'] else num_it_per_epoch // 2
    args.num_it_per_epoch = num_it_per_epoch
    saveExperimentSettings(args, os.path.join(args_dict['output_dir'], 'settings.yaml'))
    print(args_dict)

    try:
        for epoch in epoch_range:
            train_loader = get_data_loaders_acdc(training_set, args_dict, rs=rs)
            pbar = tqdm(train_loader, desc="Training AE for SR e{}".format(epoch), total=num_it_per_epoch)
            trainer.reset_losses()
            for batch_item in pbar:
                batch_item = prepare_batch_pairs(batch_item, expand_type="repeat")
                do_validate = True if (trainer.iters + 1) % num_it_per_epoch == 0 else False
                trainer.train(batch_item, keep_predictions=do_validate)
                pbar.set_description("Training AE for SR e{} loss {:.6f}".format(epoch, trainer.losses['loss_ae'][-1]))
                if do_validate:
                    val_result_dict = trainer.validate(validation_batch, image_dict=image_dict_val)
                    trainer.show_loss_on_tensorboard()
                    trainer.show_loss_on_tensorboard(eval_type="test")
                    trainer.generate_train_images(epoch=epoch, batch_item=batch_item)
                    # Enable if you want examples of synthesized validation volumes on tensorboard
                    # trainer.add_image_tensorboard(val_result_dict['img_grid_recons'], "reconstructed/test")
                    # for p_id, s_img in val_result_dict['synthesized_vols'].items():
                    #    trainer.add_image_tensorboard(s_img, "synthesized/test-pat{:03d}".format(p_id))
                    # reset training/test loss arrays after we posted averages on tensorboard
                    trainer.reset_losses()
                    # generate, save, post result of synthesized images during training (from batch)

            # End epoch: save model and do some other stuff to follow training (make images)
            trainer.end_epoch_processing(epoch=epoch, val_result_dict=val_result_dict, batch_item=batch_item)

    except KeyboardInterrupt:
        fname = os.path.join(args_dict['dir_models'], '{:0d}.models'.format(epoch))
        print("KeyboardInterrupt - Save model and exit {}".format(fname))
        trainer.save_models(fname, epoch)
    # Final save
    fname = os.path.join(args_dict['dir_models'], '{:0d}.models'.format(epoch))
    trainer.save_models(fname, epoch)


if __name__ == '__main__':
    args, args_dict = parse_args()
    model_config = NetworkConfig(args_dict['model'], dataset=args_dict['dataset'], ae_class=args_dict['ae_class'])
    args_dict = merge_args_architecture(args_dict, model_config.architecture)
    torch.manual_seed(args_dict['seed'])
    torch.cuda.manual_seed(args_dict['seed'])
    rs = np.random.RandomState(args_dict['seed'])
    main()

# python train_cardiac_aesr.py --dataset=ACDC --model=ae_combined --batch_size=12 --test_batch_size=16 --latent=128
# --latent_width=32 --width=128 --exper_id=pool2_w32_l128_aug160_w005_ex01 --downsample_steps=2 --epochs=900
# --ex_loss_weight1=0.05 --aug_patch_size=160 --epoch_threshold=500 --output_dir=/home/jorg/expers/sr_bogus
