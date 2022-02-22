import glob
import sys
import os
sys.path.extend([os.path.expanduser('~/repo/aesr')])
import numpy as np
from kwatsch.get_trainer import get_trainer
from torchvision import transforms
from datasets.shared_transforms import CenterCrop, GenericToTensor, AdjustToPatchSize
from evaluate.evaluate_interpolations import evaluate_interpolation_performance
import types
import argparse
from kwatsch.get_trainer import get_trainer_dynamic
from datasets.ACDC.data import acdc_all_image4d
from datasets.ACDC.data4d_simple import get_patids_acdc_sr
from datasets.dHCP.create_dataset import get_patient_ids as get_dHCP_patient_ids
from datasets.OASIS.dataset import get_oasis_patient_ids
from datasets.ADNI.create_dataset import get_patient_ids as get_patient_ids_adni
from datasets.MNIST.data3d import get_mnist_ids
from datasets.data_config import get_config
from evaluate.brain.evaluate_model import get_test_dataset
from kwatsch.common import loadExperimentSettings


def get_transforms(transform_patch_size, to_tensor=True):

    t = [AdjustToPatchSize((transform_patch_size, transform_patch_size)),
                                        CenterCrop(transform_patch_size)]
    if to_tensor:
        t.append(GenericToTensor())
    transform = transforms.Compose(t)

    return transform


def store_top_scores(model_nbr, top_scores, ssim_results, psnr_results, vif_results):
    mean_ssim, std_ssim = np.mean(np.array(ssim_results)), np.std(np.array(ssim_results))
    mean_psnr, std_psnr = np.mean(np.array(psnr_results)), np.std(np.array(psnr_results))
    mean_vif, std_vif = np.mean(np.array(vif_results)), np.std(np.array(vif_results))
    top_scores[model_nbr] = np.array([mean_ssim, mean_psnr, mean_vif])
    return top_scores


def find_best_val_model(data_generator, exper_src_dir, epoch_range=None, ps_evaluate=None, eval_axis=0,
                        downsample_steps=None, patient_id=None, limit_4d=False, func_get_trainer=get_trainer):
    exper_src_dir = os.path.expanduser(exper_src_dir)
    search_mask = os.path.join(os.path.join(exper_src_dir, "models"), "*.models")
    model_list = glob.glob(search_mask)
    model_list.sort()

    if epoch_range is not None:
        epoch_range = [str(e) for e in epoch_range]
        model_list = [mpath for mpath in model_list if os.path.basename(mpath).replace(".models", "") in epoch_range]
        model_list.sort()
    print("INFO - find-best-validation-model - testing {} networks using p-size {} "
          " - eval_axis={}".format(len(model_list), ps_evaluate, eval_axis))
    if len(model_list) == 0:
        raise ValueError("Error no models found with search mask {}".format(search_mask))
    if isinstance(data_generator, types.GeneratorType):
        if limit_4d:
            # for 4d data ACDC/ARVC sometimes we only want to evaluate one time frame (faster)
            data_generator = {i: t for i, t in enumerate(data_generator) if t['frame_id'] in [4, 11, 15]}
        else:
            data_generator = {i: t for i, t in enumerate(data_generator)}
        print("Transformed data generator into dict with len {}".format(len(data_generator)))
    top_scores, top_scores_synth = {}, {}
    best_ssim, best_psnr, best_vif = tuple((None, 0)), tuple((None, 0)), tuple(("None", -1))
    best_ssim_synth, best_psnr_synth, best_vif_synth = tuple((None, 0)), tuple((None, 0)), tuple(("None", -1))
    for model_nbr in epoch_range:
        trainer, e_args = func_get_trainer(src_path=exper_src_dir, model_nbr=model_nbr, eval_mode=True)
        if downsample_steps is None:
            if "downsample_steps" not in e_args.keys():
                raise ValueError("ERROR - Downsample steps need to be specified")
            downsample_steps = e_args["downsample_steps"]
        transform = get_transforms(transform_patch_size=ps_evaluate, to_tensor=False)
        result_dict = \
            evaluate_interpolation_performance(trainer, e_args, data_generator, transform=transform,
                                               downsample_steps=downsample_steps, file_suffix=None,
                                               patient_id=patient_id, eval_axis=eval_axis)
        ssim_results, psnr_results, vif_results = result_dict['ssim'], result_dict['psnr'], result_dict['vif']
        top_scores = store_top_scores(model_nbr, top_scores, ssim_results, psnr_results, vif_results)
        if top_scores[model_nbr][0] > best_ssim[1]:
            best_ssim = np.array([int(model_nbr), top_scores[model_nbr][0]]).astype(np.float32)
        if top_scores[model_nbr][1] > best_psnr[1]:
            best_psnr = np.array([int(model_nbr), top_scores[model_nbr][1]]).astype(np.float32)
        if top_scores[model_nbr][2] > best_vif[1]:
            best_vif = np.array([int(model_nbr), top_scores[model_nbr][2]]).astype(np.float32)
        ssim_results, psnr_results, vif_results = \
            result_dict['ssim_synth'], result_dict['psnr_synth'], result_dict['vif_synth']
        top_scores_synth = store_top_scores(model_nbr, top_scores_synth, ssim_results, psnr_results, vif_results)
        if top_scores_synth[model_nbr][0] > best_ssim_synth[1]:
            best_ssim_synth = np.array([int(model_nbr), top_scores_synth[model_nbr][0]]).astype(np.float32)
        if top_scores_synth[model_nbr][1] > best_psnr_synth[1]:
            best_psnr_synth = np.array([int(model_nbr), top_scores_synth[model_nbr][1]]).astype(np.float32)
        if top_scores_synth[model_nbr][2] > best_vif_synth[1]:
            best_vif_synth = np.array([int(model_nbr), top_scores_synth[model_nbr][2]]).astype(np.float32)
    print("Top metrics: Mean SSIM/PSNR/VIF M-{}: {:.4f} / M-{}: {:.4f} "
          "M-{}: {:.4f}".format(best_ssim[0], best_ssim[1], best_psnr[0], best_psnr[1],
                                best_vif[0], best_vif[1]))
    np_fname = os.path.join(exper_src_dir, "model_perf_{}_to_{}_axis{}.npz".format(epoch_range[0], epoch_range[-1],
                                                                                   eval_axis))
    np.savez(np_fname, **top_scores)
    print("Top synthesis: Mean SSIM/PSNR/VIF M-{}: {:.4f} / M-{}: {:.4f} "
          "M-{}: {:.4f}".format(best_ssim_synth[0], best_ssim_synth[1], best_psnr_synth[0], best_psnr_synth[1],
                                best_vif_synth[0], best_vif_synth[1]))
    print("Saved result dict to {}".format(np_fname))
    np_fname = os.path.join(exper_src_dir, "model_perf_synth_{}_to_{}_axis{}.npz".format(epoch_range[0], epoch_range[-1],
                                                                                   eval_axis))
    np.savez(np_fname, **top_scores_synth)
    return dict(sorted(top_scores.items()))


def load_model_scores(exper_dir, file_suffix='.npz', synthesis=False):

    load_dir = os.path.expanduser(exper_dir)
    file_prefix = "model_perf_synth*" if synthesis else "model_perf*"
    search_mask = os.path.join(load_dir, file_prefix + file_suffix)
    print("INFO - searching with mask {}".format(search_mask))
    files_to_load = glob.glob(search_mask)
    if len(files_to_load) == 0:
        print("INFO - nothing to load from {}".format(load_dir))
        return None
    results = {}
    for fname in files_to_load:
        if not synthesis:
            if 'synth' in fname:
                continue  # skip results for synthesis 
        print("INFO - loading {}".format(fname))
        np_files = np.load(fname)
        results.update({epoch: np_files[epoch] for epoch in np_files.files})

    epochs, ssim, psnr, vif = [], [], [], []
    for epoch, metrics in results.items():
        epochs.append(int(epoch)), ssim.append(metrics[0]), psnr.append(metrics[1]), vif.append(metrics[2])

    return results, np.array(epochs), np.array(ssim), np.array(psnr), np.array(vif)


def get_data_generator():
    global args

    if args.dataset == 'dHCP':
        dataset_config = get_config(args.dataset)
        pat_nums_validation = get_dHCP_patient_ids("validation", dataset_config.image_dir)
        print("Validating on {} patients".format(len(pat_nums_validation)))
        generator, _ = get_test_dataset(args.dataset, patient_id=None, downsample=False,
                                          downsample_steps=args.downsample_steps,
                                          type_of_set="validation", patch_size=args.eval_patch_size,
                                          include_hr_images=True, patid_list=pat_nums_validation)
    elif args.dataset == "OASIS":
        # ONLY USING A SUBSET OF THE VALIDATION SET FOR THIS
        pat_nums_validation = get_oasis_patient_ids("validation")[::2]
        generator, _ = get_test_dataset(args.dataset, patient_id=None, downsample=False,
                                        downsample_steps=args.downsample_steps,
                                        type_of_set="validation", patch_size=args.eval_patch_size,
                                        include_hr_images=True, patid_list=pat_nums_validation)
    elif args.dataset == "ADNI":
        # ONLY USING A SUBSET OF THE VALIDATION SET FOR THIS
        pat_nums_validation = get_patient_ids_adni("validation")[:10]
        generator, _ = get_test_dataset(args.dataset, patient_id=None, downsample=False,
                                        downsample_steps=args.downsample_steps,
                                        type_of_set="validation", patch_size=args.eval_patch_size,
                                        include_hr_images=True, patid_list=pat_nums_validation)
    elif args.dataset == 'ACDC':
        pat_nums_validation = get_patids_acdc_sr("validation", rs=rs, limited_load=False)
        pat_nums_validation.sort()
        print("Validating on {} patients".format(len(pat_nums_validation)))
        generator = acdc_all_image4d(os.path.expanduser('~/data/ACDC/all_cardiac_phases'), resample=True, rescale=True,
                                          new_spacing=tuple((1, 1.4, 1.4)),
                                          limited_load=False, patid_list=pat_nums_validation[:10])
    elif args.dataset in ["MNIST3D", 'MNIST', 'MNISTRoto']:
        # Important: for MNISTRoto we use same synthetic volumes as for MNIST3D
        dataset = 'MNIST3D' if args.dataset == 'MNISTRoto' else args.dataset
        pat_nums_validation = get_mnist_ids('validation')[::10]
        pat_nums_validation.sort()
        generator, _ = get_test_dataset(dataset, patient_id=None, downsample=False,
                                        downsample_steps=args.downsample_steps,
                                        type_of_set="test", patch_size=args.eval_patch_size,
                                        include_hr_images=True, patid_list=pat_nums_validation)

    print("Validating on {} patients".format(len(pat_nums_validation)))
    return generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find best SR model')

    parser.add_argument('--epoch_range', type=int, nargs=2, default=[200, 201])
    parser.add_argument('--exper_dir', type=str, default=None)
    parser.add_argument('--eval_patch_size', type=int, default=None)
    parser.add_argument('--eval_axis', type=int, default=0)
    parser.add_argument('--downsample_steps', type=int, default=None)
    parser.add_argument('--dataset', type=str, choices=['ACDC', 'dHCP', 'OASIS', 'MNISTRoto', 'ADNI',
                                                        'None'], default='None')
    args = parser.parse_args()
    args.exper_dir = os.path.expanduser(args.exper_dir)
    setting_file = os.path.join(args.exper_dir, "settings.yaml")
    original_args = loadExperimentSettings(setting_file)
    if args.downsample_steps is None:
        args.downsample_steps = original_args.downsample_steps
        print("Warning - using downsampling steps {}".format(args.downsample_steps))
    if args.dataset == 'None':
        args.dataset = original_args.dataset
    print(args)
    rs = np.random.RandomState(32563)
    epoch_range = np.arange(args.epoch_range[0], args.epoch_range[1] + 1)
    patient_id = None  #
    data_generator = get_data_generator()
    result_dict = find_best_val_model(data_generator,
                                      args.exper_dir,
                                      epoch_range, ps_evaluate=args.eval_patch_size,
                                      downsample_steps=args.downsample_steps, patient_id=patient_id,
                                      limit_4d=True, func_get_trainer=get_trainer_dynamic,
                                      eval_axis=args.eval_axis)