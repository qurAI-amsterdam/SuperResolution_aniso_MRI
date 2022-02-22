import math
import torch
import torch.nn as nn
import numpy as np

# norm_layer = partial(nn.InstanceNorm2d, affine=True)
norm_layer = nn.BatchNorm2d
activation = nn.LeakyReLU


# authors use this initializer, but it doesn't seem essential
def Initializer(layers, slope=0.2):
    for layer in layers:
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            std = 1 / np.sqrt((1 + slope ** 2) * np.prod(w.shape[:-1]))
            w.normal_(std=std)
        if hasattr(layer, 'bias'):
            layer.bias.data.zero_()


def Encoder(scales, depth, latent, colors, n_res_block=None, use_batchnorm=False):

    layers = []
    layers.append(nn.Conv2d(colors, depth // 2, 1, padding=1))
    kp = depth // 2
    for scale in range(scales):
        k = depth << scale
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
        if use_batchnorm:
            layers.extend([nn.BatchNorm2d(k)])
        layers.append(nn.AvgPool2d(2))
        kp = k

    k = depth << scales
    layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])

    layers.append(nn.Conv2d(k, latent, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)


def Decoder(scales, depth, latent, colors, n_res_block=None, use_upsample=True, use_batchnorm=False):
    layers = []
    channels_layer1 = depth << scales
    layers.extend([nn.Conv2d(latent, channels_layer1, 1, padding=0), activation()])
    if use_batchnorm:
        layers.extend([nn.BatchNorm2d(channels_layer1)])
    kp = channels_layer1
    for scale in range(scales - 1, -1, -1):
        k = depth << scale
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
        if use_batchnorm:
            layers.extend([nn.BatchNorm2d(k)])
        if use_upsample:
            layers.append(nn.Upsample(scale_factor=2))
        else:
            layers.append(nn.ConvTranspose2d(k, k, 4, stride=2, padding=1))
        kp = k
    layers.extend([nn.Conv2d(kp, depth, 3, padding=1), activation()])
    Initializer(layers)
    return nn.Sequential(*layers)


class MultiChannelAE(nn.Module):

    def __init__(self, args):
        super().__init__()
        scales = int(round(math.log(args['width'] // args['latent_width'], 2)))
        if "n_res_block" not in args.keys():
            args['n_res_block'] = None
        if 'use_batchnorm' not in args.keys():
            args['use_batchnorm'] = False
        if 'use_sigmoid' not in args.keys():
            args['use_sigmoid'] = False
        if 'gpu_ids' not in args.keys():
            args['gpu_ids'] = [0]
        self.enc = Encoder(scales, args['depth'], args['latent'], args['colors'],
                           n_res_block=args['n_res_block'], use_batchnorm=args['use_batchnorm']).to(args['device'])
        self.dec = Decoder(scales, args['depth'], args['latent'], args['colors'],
                           n_res_block=args['n_res_block'], use_batchnorm=args['use_batchnorm']).to(args['device'])

        self.decoder_head1 = [nn.Conv2d(args['depth'], 1, 3, padding=1), nn.Sigmoid()]
        self.decoder_head2 = [nn.Conv2d(args['depth'], args['depth'], 3, padding=1), activation(), nn.BatchNorm2d(args['depth']),
                              nn.Conv2d(args['depth'], args['nclasses'], 1, padding=0), nn.Softmax(dim=1)]
        Initializer(self.decoder_head1)
        Initializer(self.decoder_head2)
        self.decoder_head1 = nn.Sequential(*self.decoder_head1)
        self.decoder_head2 = nn.Sequential(*self.decoder_head2)

    def forward(self, img):
        return self.decode(self.encode(img))

    def encode(self, img):
        return self.enc(img)

    def decode(self, z):
        out = self.dec(z)
        img = self.decoder_head1(out)
        soft_probs = self.decoder_head2(out)
        return {'image': img, 'soft_probs': soft_probs}


if __name__ == "__main__":
    args = {
        'dataset': 'ACDC',  # 'mnist'   'synthetic'   'acdc_labels'
        'model': 'aesr',  # acai
        'epochs': 10,
        'width': 128,
        'latent_width': 32,  # 4
        'depth': 64,  # 16
        'use_batchnorm': True,
        'advdepth': 16,
        'advweight': 0.5,
        'reg': 0.2,
        'latent': 128,  # was 4=mnist32
        'colors': 2,
        'nclasses': 4,
        'lr': 0.00001,
        'batch_size': 16,
        'test_batch_size': 16,
        'device': 'cuda',
        'seed': 32563,
        'weight_decay': 0,  # 1e-5
        'limited_load': False,
        'fold': 0,
        'use_laploss': True,
        'n_res_block': None,
        'use_percept_loss': False,
        'src_data_path': "~/data/BrainMASI_LR_co"
    }

    ae_model = MultiChannelAE(args).to(args['device'])
    # print(ae_model)
    a = torch.randn(16, 1, args['width'], args['width']).to(args['device'])
    b = torch.randn(16, 1, args['width'], args['width']).to(args['device'])
    out = ae_model(torch.cat([a, b], dim=1))
    print(ae_model)
    scales = int(round(math.log(args['width'] // args['latent_width'], 2)))
    print("#Scales {}".format(scales))
    print("out.shape ", out['image'].shape, out['labels'].shape)
