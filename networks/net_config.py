
MODULE_PATH = {"VanillaACAI": "networks/acai_vanilla.py",
               'VAE': "networks/beta_vae.py",
               'VAE2': "networks/beta_vae.py",
               "LargerAE": "networks/acai_vanilla_modified.py",
               "MultiChannelAE": "networks/acai_multi_channel.py",
               "VanillaACAIStrided": "networks/acai_vanilla_strided.py"}


class NetworkConfig(object):

    def __init__(self, network, dataset=None, ae_class="VanillaACAI"):
        self.network = network
        self.dataset = dataset
        self.ae_class = ae_class
        self.architecture = {}
        self.load_config()

    def load_config(self):
        self.architecture['width'] = 128
        self.architecture['latent_width'] = 16
        self.architecture['depth'] = 32
        self.architecture['colors'] = 2 if self.dataset == 'ACDCLBL' else 1
        self.architecture['latent'] = 16
        self.architecture['use_laploss'] = False
        self.architecture['use_percept_loss'] = False
        self.architecture['n_res_block'] = None
        self.architecture['use_batchnorm'] = True
        self.architecture['use_sigmoid'] = True
        self.architecture['max_grad_norm'] = 0
        self.architecture['fine_tune'] = False
        self.architecture['ex_loss_weight1'] = 0.5
        self.architecture['module_network_path'] = MODULE_PATH[self.ae_class]

        if self.network == "ae" or self.network == "aesr":
            if self.dataset is None or self.dataset == "ACDC":
                self.architecture['module_trainer_path'] = "kwatsch/trainer_ae.py"
                self.architecture['trainer_class'] = "AEBaseTrainer"
            elif self.dataset == "ACDCLBL":
                self.architecture['module_trainer_path'] = "kwatsch/sr_multi_channel/trainer_ae.py"
                self.architecture['trainer_class'] = "MultiChannelTrainer"
                self.architecture['nclasses'] = 4
            elif self.dataset in ['dHCP', 'ADNI', 'OASIS']:
                self.architecture['module_trainer_path'] = "kwatsch/brain/trainer_ae.py"
                self.architecture['trainer_class'] = "AETrainerBrain"
            elif self.dataset in ["MNIST3D", 'MNISTRoto']:
                self.architecture['image_mix_loss_func'] = "perceptual"
                self.architecture['module_trainer_path'] = "kwatsch/mnist/trainer_ae.py"
                self.architecture['trainer_class'] = "AETrainerMNIST"
            else:
                raise ValueError("Error - NetworkConfig - Unsupported combination {}/{}".format(self.network,
                                                                                                self.dataset))
            self.architecture['image_mix_loss_func'] = None
        elif self.network == "ae_combined" or self.network == "aesr_combined":
            self.architecture['image_mix_loss_func'] = "perceptual"
            if self.dataset == "ACDC":
                self.architecture['module_trainer_path'] = "kwatsch/cardiac/trainer_ae.py"
                self.architecture['trainer_class'] = "AETrainerEndToEnd"
            elif self.dataset == "ACDCLBL":
                self.architecture['module_trainer_path'] = "kwatsch/sr_multi_channel/trainer_ae.py"
                self.architecture['trainer_class'] = "MultiChannelCAISRTrainer"
                self.architecture['nclasses'] = 4
            elif self.dataset in ['dHCP', 'ADNI', 'OASIS']:
                self.architecture['module_trainer_path'] = "kwatsch/brain/trainer_ae.py"
                self.architecture['trainer_class'] = "AETrainerExtension1Brain"
            elif self.dataset in ["MNIST3D", 'MNISTRoto']:
                self.architecture['module_trainer_path'] = "kwatsch/mnist/trainer_ae.py"
                self.architecture['trainer_class'] = "AECombinedTrainerMNIST"

            else:
                raise ValueError("Error - NetworkConfig - Unsupported combination {}/{}".format(self.network,
                                                                                                self.dataset))
        elif self.network in ['vae', 'vae_combined', 'vae2']:
            if self.dataset in ["MNIST3D", 'MNISTRoto', 'ACDC', 'OASIS', 'dHCP', 'ADNI']:
                if 'combined' in self.network:
                    self.architecture['image_mix_loss_func'] = "perceptual"
                else:
                    self.architecture['image_mix_loss_func'] = None
                self.architecture['module_trainer_path'] = "kwatsch/trainer_vae.py"
                self.architecture['trainer_class'] = "VAETrainer"
            else:
                raise ValueError("Error - network VAE does not support dataset {}".format(self.dataset))
        elif self.network in ['acai', 'acai_combined']:
            if self.dataset in ["MNIST3D", 'MNISTRoto', 'ACDC', 'OASIS', 'dHCP', 'ADNI']:
                if 'combined' in self.network:
                    self.architecture['image_mix_loss_func'] = "perceptual"
                else:
                    self.architecture['image_mix_loss_func'] = None
                self.architecture['module_trainer_path'] = "kwatsch/trainer_acai.py"
                self.architecture['trainer_class'] = "ACAITrainer"
            else:
                raise ValueError("Error - network ACAI does not support dataset {}".format(self.dataset))

