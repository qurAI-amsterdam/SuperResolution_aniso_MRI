import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from collections import defaultdict
from kwatsch.acai_utils import clip_grad_norm
from kwatsch.lap_pyramid_loss import LapLoss
from lpips.perceptual import PerceptualLoss
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

from functools import partial
# norm_layer = partial(nn.InstanceNorm2d, affine=True)
norm_layer = nn.BatchNorm2d
activation = nn.LeakyReLU


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


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
    layers.append(nn.Conv2d(colors, depth, 1, padding=1))
    kp = depth
    for scale in range(scales):
        k = depth << scale
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
        if use_batchnorm:
            layers.extend([nn.BatchNorm2d(k)])
        layers.append(nn.AvgPool2d(2))
        kp = k

    if n_res_block is not None:
        for i in range(n_res_block):
            layers.append(ResBlock(kp, channel=128))
        layers.append(nn.ReLU(inplace=True))

    k = depth << scales
    layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])

    layers.append(nn.Conv2d(k, latent, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)


def Decoder(scales, depth, latent, colors, n_res_block=None, use_upsample=True, use_batchnorm=False,
            use_sigmoid=False):
    layers = []
    if n_res_block is not None:
        for i in range(n_res_block):
            layers.append(ResBlock(latent, channel=128))

        layers.append(nn.ReLU(inplace=True))

    kp = latent
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
    if use_sigmoid:
        layers.extend([nn.Conv2d(depth, colors, 3, padding=1), nn.Sigmoid()])
    else:
        layers.append(nn.Conv2d(depth, colors, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)


def create_decoder(args):
    scales = int(round(math.log(args['width'] // args['latent_width'], 2)))
    return Decoder(scales, args['depth'], args['latent'], args['colors'],
                       n_res_block=args['n_res_block'], use_batchnorm=args['use_batchnorm'],
                       use_sigmoid=args['use_sigmoid']).to(args['device'])


class VanillaACAI(nn.Module):

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
                           n_res_block=args['n_res_block'], use_batchnorm=args['use_batchnorm'],
                           use_sigmoid=args['use_sigmoid']).to(args['device'])

    def forward(self, img):
        return self.decode(self.encode(img))

    def encode(self, img):
        return self.enc(img)

    def decode(self, z):
        return self.dec(z)


class Discriminator(nn.Module):

    def __init__(self, args):
        super().__init__()
        scales = int(round(math.log(args['width'] // args['latent_width'], 2)))
        self.use_sigmoid = False  # args['use_sigmoid']
        self.encoder = Encoder(scales, args['depth'], args['latent'], args['colors'],
                           n_res_block=args['n_res_block'], use_batchnorm=args['use_batchnorm']).to(args['device'])

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.mean(x, -1)
        if self.use_sigmoid:
            return torch.sigmoid(x)
        else:
            return x


def swap_halves(x):
    a, b = x.split(x.shape[0]//2)
    return torch.cat([b, a])


# torch.lerp only support scalar weight
def lerp(start, end, weights):
    return start + weights * (end - start)


def L2(x):
    return torch.mean(x**2)


if __name__ == "__main__":
    args = {
        'dataset': 'brainMASI',  # 'mnist'   'synthetic'   'acdc_labels'
        'model': 'aesr',  # acai
        'epochs': 10,
        'width': 128,
        'latent_width': 32,  # 4
        'depth': 32,  # 16
        'advdepth': 16,
        'advweight': 0.5,
        'reg': 0.2,
        'latent': 128,  # was 4=mnist32
        'colors': 1,
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

    ae_model = VanillaACAI(args).to(args['device'])
    # print(ae_model)
    x = torch.randn(16, 1, args['width'], args['width']).to(args['device'])
    out = ae_model(x)
    print(ae_model)
    scales = int(round(math.log(args['width'] // args['latent_width'], 2)))
    print("#Scales {}".format(scales))
    print("Shape output ", out.detach().cpu().numpy().shape)