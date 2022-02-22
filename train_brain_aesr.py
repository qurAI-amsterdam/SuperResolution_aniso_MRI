import os
from tqdm import tqdm
from kwatsch.common import saveExperimentSettings
from shutil import copytree, rmtree
import torch
import numpy as np
from kwatsch.arguments import parse_args
from kwatsch.get_trainer import get_trainer_dynamic
from networks.net_config import NetworkConfig

from datasets.brainMASI.dataset import get_dataset_brainMASI
from datasets.dHCP.dataset import get_dataset_braindHCP
from datasets.OASIS.dataset import get_dataset_brainOASIS
from datasets.ADNI.dataset import get_dataset_brainADNI
from datasets.MNIST.data3d import get_dataset_MNIST3D
from datasets.MNIST.data_roto import get_dataset_MNISTRoto
from datasets.common_brains import get_data_loaders_brain_dataset, get_transforms_brain
from datasets.common_brains import prepare_batch_pairs as prepare_batch_pairs_brain


def merge_args_architecture(args_dict, architecture):
    for key, value in architecture.items():
        # arguments passed on command line supersede config loaded from architecture (net_config.py)
        # BUT ONLY, if argments are not NONE. Boolean argments are False by default and hence, need to be passed
        # on command line to supersede architecture. This is important e.g. use_sigmoid!
        if key not in args_dict.keys() or (args_dict[key] is None):
            args_dict[key] = value
    return args_dict


def get_datasets_brain(transform_tr, transform_te):
    global args_dict, rs
    args_dict['downsample'] = False if args_dict['downsample_steps'] is None else True
    if args_dict['dataset'] == 'brainMASI':
        args_dict['src_data_path'] = os.path.expanduser("~/data/brainMASI_cropped/")
        training_set, val_set = get_dataset_brainMASI(args_dict, args_dict['src_data_path'], rs,
                                                      type_of_set="both",
                                                      downsample=args_dict['downsample'],
                                                      downsample_steps=args_dict['downsample_steps'],
                                                      transform_tr=transform_tr,
                                                      transform_te=transform_te, test_limited_load=True)
    elif args_dict['dataset'] == 'dHCP':
        args_dict['src_data_path'] = os.path.expanduser("~/data/dHCP_cropped_256/")
        training_set, val_set = get_dataset_braindHCP(args_dict, args_dict['src_data_path'], rs,
                                                      type_of_set="both",
                                                      downsample=args_dict['downsample'],
                                                      downsample_steps=args_dict['downsample_steps'],
                                                      transform_tr=transform_tr,
                                                      transform_te=transform_te, test_limited_load=True)
    elif args_dict['dataset'] == 'OASIS':
        args_dict['src_data_path'] = os.path.expanduser("~/data/OASIS/nifti")
        training_set, val_set = get_dataset_brainOASIS(args_dict, args_dict['src_data_path'], rs,
                                                      type_of_set="both",
                                                      downsample=args_dict['downsample'],
                                                      downsample_steps=args_dict['downsample_steps'],
                                                      transform_tr=transform_tr,
                                                      transform_te=transform_te, test_limited_load=True)
    elif args_dict['dataset'] == 'ADNI':
        args_dict['src_data_path'] = os.path.expanduser("~/data/ADNI/")
        training_set, val_set = get_dataset_brainADNI(args_dict, args_dict['src_data_path'], rs,
                                                      type_of_set="both",
                                                      downsample=args_dict['downsample'],
                                                      downsample_steps=args_dict['downsample_steps'],
                                                      transform_tr=transform_tr,
                                                      transform_te=transform_te, test_limited_load=True)

    elif args_dict['dataset'] == 'MNIST3D':
        args_dict['src_data_path'] = os.path.expanduser("~/data/MNIST3D")
        training_set, val_set = get_dataset_MNIST3D(args_dict, args_dict['src_data_path'],
                                                    type_of_set="both",
                                                    downsample=args_dict['downsample'],
                                                    downsample_steps=args_dict['downsample_steps'],
                                                    transform_tr=transform_tr,
                                                    transform_te=transform_te, test_limited_load=True)
    elif args_dict['dataset'] == 'MNISTRoto':
        args_dict['src_data_path'] = os.path.expanduser("~/data/")
        training_set, val_set = get_dataset_MNISTRoto(args_dict, args_dict['src_data_path'],
                                                    type_of_set="both",
                                                    downsample=args_dict['downsample'],
                                                    downsample_steps=args_dict['downsample_steps'],
                                                    transform_tr=transform_tr,
                                                    transform_te=transform_te, test_limited_load=True)
    else:
        raise ValueError("Error - get-datasets - Unknown dataset {}".format(args_dict['dataset']))

    return training_set, val_set


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
    if args_dict['dataset'] in ['dHCP', 'brainMASI']:
        args_dict['get_masks'] = False
    print("INFO - train AE for SR - using output dir {}".format(args_dict['output_dir']))


def prepare_validation_batch(test_loader):
    global args_dict
    validation_batch = test_loader.next()
    validation_batch = prepare_batch_pairs_brain(validation_batch, expand_type="repeat")
    validation_batch['image'] = validation_batch['image'].to(args_dict['device'])
    if 'loss_mask' in validation_batch.keys():
        validation_batch['loss_mask'] = validation_batch['loss_mask'].float()
    return validation_batch


def main():
    global args, args_dict, rs
    prepare_run()
    transform_tr, transform_te = get_transforms_brain(args_dict['dataset'], rs=rs, patch_size=args_dict['width'],
                                                      aug_patch_size=args_dict['aug_patch_size'])
    training_set, val_set = get_datasets_brain(transform_tr, transform_te)
    train_loader = get_data_loaders_brain_dataset(training_set, args_dict)
    test_loader = get_data_loaders_brain_dataset(val_set, args_dict, shuffle=True)
    test_loader = iter(test_loader)
    validation_batch = prepare_validation_batch(test_loader)
    trainer = get_trainer_dynamic(args_dict, model_file=args_dict['model_filename'])
    trainer.init_tensorboard(args_dict['output_dir'])
    epoch_range = generate_epoch_range()
    num_train_samples = len(train_loader.dataset)
    num_it_per_epoch = num_train_samples // args_dict['batch_size']
    print("INFO - {} - num_train_samples/batch-size/num_it_per_epoch: {} {} {}"
          "".format(args.dataset, num_train_samples, args.batch_size, num_it_per_epoch))
    show_every = args_dict['validate_every'] if num_it_per_epoch > args_dict['validate_every'] else num_it_per_epoch
    print(args_dict)
    try:
        for epoch in epoch_range:
            train_loader = get_data_loaders_brain_dataset(training_set, args_dict)
            pbar = tqdm(train_loader, desc="Training AE for SR e{}".format(epoch),
                                   total=num_it_per_epoch)
            trainer.reset_losses()
            for batch_item in pbar:
                batch_item = prepare_batch_pairs_brain(batch_item, expand_type="repeat")
                do_validate = True if (trainer.iters + 1) % num_it_per_epoch == 0 else False
                trainer.train(batch_item, keep_predictions=do_validate)
                pbar.set_description("Training AE for SR e{} loss {:.6f}".format(epoch, trainer.losses['loss_ae'][-1]))
                if do_validate:
                    #
                    val_result_dict = trainer.validate(validation_batch, image_dict=None)
                    trainer.show_loss_on_tensorboard()
                    trainer.show_loss_on_tensorboard(eval_type="test")
                    trainer.generate_train_images(epoch=epoch, batch_item=batch_item)
                    # Enable if you want examples of synthesized validation volumes on tensorboard
                    # trainer.add_image_tensorboard(val_result_dict['img_grid_recons'], "reconstructed/test")
                    # for p_id, s_img in val_result_dict['synthesized_vols'].items():
                    #    trainer.add_image_tensorboard(s_img, "synthesized/test-pat{:03d}".format(p_id))
                    # reset training/test loss arrays after we posted averages on tensorboard
                    trainer.reset_losses()
                # Disabled: saving model and images to frequently
                # if num_it_per_epoch > 10000 and trainer.iters % 10000 == 0 and trainer.iters > 1:
                #     trainer.save_model(epoch=epoch, with_iters=True)
                #     trainer.save_losses()
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

"""
    python train_aesr.py --dataset=OASIS --model=ae_combined --batch_size=8 --test_batch_size=64 --width=220 
    --latent_width=55 --latent=16 --downsample_steps=5 --epochs=1000 --image_mix_loss_func=perceptual --exper_id=
"""