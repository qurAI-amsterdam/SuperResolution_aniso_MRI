from kwatsch.common import load_settings
from networks.acai_vanilla import VanillaACAI
from kwatsch.trainer_ae import AEBaseTrainer
from kwatsch.cardiac.trainer_ae import AETrainerEndToEnd
from kwatsch.alpha.trainer_alpha import AlphaTrainer, AlphaTrainerEndToEnd
from kwatsch.alpha.trainer_alpha_only import AlphaOnlyTrainer
from kwatsch.alpha.alpha_combined import AlphaTrainerCombined
from networks.acai_vanilla import create_decoder
import networks.alpha.alpha_network
from importlib import import_module
import os


def syth_src_path(fname):
    src_path = "~" if "~" in fname else "/"
    parts = fname.split(os.sep)
    for p in parts[:-1]:
        if "model" not in p and '~' not in p:
            src_path = os.path.join(src_path, p)
    return os.path.expanduser(src_path)


def get_trainer_dynamic(args_dict=None, src_path=None, model_nbr=None, eval_mode=False, args_only=False,
                        model_file=None, **kwargs):
    """

        Args:
            args_dict:
            src_path:   pass for evaluation
            model_nbr:  pass for evaluation
            eval_mode:
            args_only:
            model_file: pass for re-training only !!!

        Returns:

        """
    if model_nbr is None and args_dict is None:
        raise ValueError("ERROR - get_trainer - args_dict or model_filename needs to be specified")
    if model_file is not None:
        print("Warning - get trainer - RETRAIN model {}".format(model_file))
    if src_path is not None:
        src_path = os.path.expanduser(src_path)
        setting_file = os.path.join(src_path, "settings.yaml")
        args_dict = load_settings(setting_file)
        if 'output_dir' not in args_dict.keys():
            args_dict['output_dir'] = src_path
        model_filename = os.path.join("models", model_nbr + ".models" if isinstance(model_nbr, str) else str(model_nbr) + ".models")
        model_file = os.path.expanduser(os.path.join(src_path, model_filename))
        model_nbr_sr = kwargs.get("model_nbr_sr", None)
        if model_nbr_sr is not None:
            model_file_sr = os.path.join("models", str(model_nbr_sr) + ".models")
            model_file_sr = os.path.expanduser(os.path.join(src_path, model_file_sr))
        else:
            model_file_sr = None

    else:
        # we are not evaluating. Hence setting our extra aesr model to None
        aesr_model, model_file_sr = None, None

    ae_class_name = "VanillaACAI" if 'ae_class' not in args_dict.keys() else args_dict['ae_class'].replace('default', 'VanillaACAI')
    args_dict['use_extra_latent_loss'] = False if "use_extra_latent_loss" not in args_dict.keys() else args_dict['use_extra_latent_loss']
    args_dict['use_alpha_probe'] = False if 'use_alpha_probe' not in args_dict.keys() else args_dict['use_alpha_probe']
    args_dict['alpha_dims'] = None if 'alpha_dims' not in args_dict.keys() else args_dict['alpha_dims']
    if args_only:
        return None, args_dict
    ae_module = import_module(args_dict['module_network_path'].replace("/", ".").replace(".py", ""))
    ae_class = getattr(ae_module, ae_class_name)
    ae_model = ae_class(args_dict).to(args_dict['device'])
    aesr_model = None if model_file_sr is None else ae_class(args_dict).to(args_dict['device'])
    trainer_module = args_dict['module_trainer_path'].replace("/", ".").replace(".py", "")
    # this is for backward compatibility, renamed module "utils" to "kwatsch"
    trainer_module = trainer_module.replace("utils.", "kwatsch.")
    trainer_module = import_module(trainer_module)
    trainer_class_name = "AEBaseTrainer" if 'trainer_class' not in args_dict.keys() else args_dict['trainer_class']
    trainer_class_name = getattr(trainer_module, trainer_class_name)
    trainer = trainer_class_name(args_dict, ae_model, model_file=model_file, eval_mode=eval_mode,
                                 model_sr=aesr_model, model_file_sr=model_file_sr)
    print("WARNING WARNING - you are using {} CAE class".format(ae_model.__class__.__name__))
    if src_path is None:
        # args_dict was passed to function
        return trainer
    else:
        # we load model from file
        return trainer, args_dict


def get_trainer(args_dict=None, src_path=None, model_nbr=None, eval_mode=False, args_only=False, model_file=None):
    """

    Args:
        args_dict:
        src_path:   pass for evaluation
        model_nbr:  pass for evaluation
        eval_mode:
        args_only:
        model_file: pass for re-training only !!!

    Returns:

    """
    if model_nbr is None and args_dict is None:
        raise ValueError("ERROR - get_trainer - args_dict or model_filename needs to be specified")
    if model_file is not None:
        print("Warning - get trainer - RETRAIN model {}".format(model_file))
    if src_path is not None:
        src_path = os.path.expanduser(src_path)
        setting_file = os.path.join(src_path, "settings.yaml")
        args_dict = load_settings(setting_file)
        if 'output_dir' not in args_dict.keys():
            args_dict['output_dir'] = src_path
        model_filename = os.path.join("models", str(model_nbr) + ".models")
        model_file = os.path.expanduser(os.path.join(src_path, model_filename))

    args_dict['inbetween_loss'] = False if 'inbetween_loss' not in args_dict.keys() else args_dict['inbetween_loss']
    args_dict['use_alpha_probe'] = False if 'use_alpha_probe' not in args_dict.keys() else args_dict['use_alpha_probe']
    args_dict['alpha_dims'] = None if 'alpha_dims' not in args_dict.keys() else args_dict['alpha_dims']
    if args_only:
        return None, args_dict

    if args_dict['model'].lower() == "ae":
        ae_model = VanillaACAI(args_dict).to(args_dict['device'])
        trainer = AEBaseTrainer(args_dict, ae_model, model_file=model_file, eval_mode=eval_mode)
    elif args_dict['model'].lower() == "ae_combined":
        if 'ae_class' not in args_dict.keys() or args_dict['ae_class'] == "default":
            ae_model = VanillaACAI(args_dict).to(args_dict['device'])
        else:
            print("INFO - get-trainer - used ExtendedACAI as autoencoder")
            # ae_model = ExtendedACAI(args_dict).to(args_dict['device'])
            raise NotImplementedError("ERROR - kwatsch.get_trainer - deprecated function yes dynamic!")
        trainer = AETrainerEndToEnd(args_dict, ae_model, model_file=model_file, eval_mode=eval_mode)
    elif args_dict['model'] == 'alpha':
        ae_model = VanillaACAI(args_dict).to(args_dict['device'])
        probe_class = getattr(networks.alpha.alpha_network, args_dict['alpha_class'])
        alpha_probe = probe_class(args_dict, additional_dims=args_dict['alpha_dims']).to(args_dict['device'])
        print("INFO - get-trainer - {} - loading alpha class {}".format(args_dict['model'], args_dict['alpha_class']))
        trainer = AlphaTrainer(args_dict, ae_model, alpha_probe=alpha_probe, model_file=model_file, eval_mode=eval_mode)
    elif args_dict['model'] == "alpha_combined":
        ae_model = VanillaACAI(args_dict).to(args_dict['device'])
        probe_class = getattr(networks.alpha.alpha_network, args_dict['alpha_class'])
        alpha_probe = probe_class(args_dict, additional_dims=args_dict['alpha_dims']).to(args_dict['device'])
        print("INFO - get-trainer - {} - loading alpha class {}".format(args_dict['model'], args_dict['alpha_class']))
        trainer = AlphaTrainerEndToEnd(args_dict, ae_model, alpha_probe=alpha_probe, model_file=model_file, eval_mode=eval_mode)
    elif args_dict['model'] == "alpha_combined_v2":
        ae_model = VanillaACAI(args_dict).to(args_dict['device'])
        probe_class = getattr(networks.alpha.alpha_network, args_dict['alpha_class'])
        alpha_probe = probe_class(args_dict, additional_dims=args_dict['alpha_dims']).to(args_dict['device'])
        decoder_mix = create_decoder(args_dict)
        print("INFO - get-trainer - {} - loading alpha class {}"
              " & 2nd decoder_mix".format(args_dict['model'], args_dict['alpha_class']))
        trainer = AlphaTrainerCombined(args_dict, ae_model, alpha_probe=alpha_probe, decoder_mix=decoder_mix,
                                       model_file=model_file, eval_mode=eval_mode)
    elif args_dict['model'] == 'alpha_only':
        src_path_ae = syth_src_path(args_dict['model_file_ae'])
        setting_file = os.path.join(src_path_ae, "settings.yaml")
        ae_args_dict = load_settings(setting_file)
        ae_model = VanillaACAI(ae_args_dict).to(args_dict['device'])
        probe_class = getattr(networks.alpha.alpha_network, args_dict['alpha_class'])
        alpha_probe = probe_class(args_dict, additional_dims=args_dict['alpha_dims']).to(args_dict['device'])
        print("INFO - get-trainer - {} - loading alpha class {}".format(args_dict['model'], args_dict['alpha_class']))
        trainer = AlphaOnlyTrainer(args_dict, ae_model, alpha_probe=alpha_probe, model_file=model_file, eval_mode=eval_mode)
    # elif args_dict['model'].lower() == "acai":
    #     # IMPORTANT: I experimented with batchnorm in discriminator. Actually one model uses
    #     # it: acdc_p128_l16_32_disc32batchnn and p128_l16_32_noaugs
    #     # In that case loading disc weights will fail. use_batchnorm needs to be equal to True
    #     args_dict['use_batchnorm_disc'] = False if not 'use_batchnorm_disc' in args_dict.keys() else args_dict['use_batchnorm_disc']
    #
    #     discriminator = Discriminator(scales, args_dict['advdepth'], args_dict['latent'],
    #                                       args_dict['colors'],
    #                                       use_batchnorm=args_dict['use_batchnorm_disc']).to(disc_gpu)  # disc_gpu
    #
    #     ae_model = VanillaACAI(args_dict).to(args_dict['device'])
    #     trainer = ACAITrainer(args_dict, ae_model, discriminator, model_file=model_file, eval_mode=eval_mode)
    else:
        raise ValueError("Error - get trainer - no trainer available for model {}".format(args_dict['model']))
    if src_path is None:
        # args_dict was passed to function
        return trainer
    else:
        # we load model from file
        return trainer, args_dict
