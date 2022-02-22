import math
import torch
import torch.nn as nn
from networks.acai_vanilla import Initializer, Decoder

activation = nn.LeakyReLU


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
        layers.append(nn.Conv2d(k, k, 2, stride=2, padding=0))
        kp = k

    k = depth << scales
    layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
    layers.append(nn.Conv2d(k, latent, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)


class VanillaACAIStrided(nn.Module):

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