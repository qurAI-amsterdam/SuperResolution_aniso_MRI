import os
import torch
import torch.nn as nn
import math
from networks.acai_vanilla import Decoder
from networks.acai_vanilla import VanillaACAI
from kwatsch.common import load_settings

norm_layer = nn.BatchNorm2d


class AlphaProbe(nn.Module):

    def __init__(self, args, additional_dims=0):
        super().__init__()
        self.use_batchnorm = False if "use_batchnorm_probe" not in args.keys() else args['use_batchnorm_probe']
        self.compress_z = [nn.Conv2d(args['latent'] * 2, 1, 1, padding=0), nn.LeakyReLU()]
        if self.use_batchnorm:
            self.compress_z.extend([nn.BatchNorm2d(1)])
        self.compress_z.extend([nn.Flatten()])
        self.compress_z = nn.Sequential(*self.compress_z)
        flattend_dim = args['latent_width'] ** 2
        self.pred_alpha = nn.Linear(flattend_dim + additional_dims, 2, bias=False)

    def forward(self, z, add_features):
        z_compressed = self.compress_z(z)
        z_compressed = torch.cat([z_compressed, add_features], dim=1)
        return self.pred_alpha(z_compressed)


class AlphaProbev2(nn.Module):

    def __init__(self, args, additional_dims=0):
        super().__init__()
        self.use_batchnorm = False if "use_batchnorm_probe" not in args.keys() else args['use_batchnorm_probe']
        self.compress_z = [nn.Conv2d(args['latent'] * 2, 1, 1, padding=0), nn.LeakyReLU()]
        if self.use_batchnorm:
            self.compress_z.extend([nn.BatchNorm2d(1)])
        self.compress_z.extend([nn.Flatten()])
        self.compress_z = nn.Sequential(*self.compress_z)
        flattend_dim = args['latent_width'] ** 2
        self.pred_alpha = nn.Sequential(*[nn.Linear(flattend_dim + additional_dims, 512), nn.ReLU(),
                                          nn.Linear(512, 2, bias=False)])

    def forward(self, z, add_features):
        z_compressed = self.compress_z(z)
        z_compressed = torch.cat([z_compressed, add_features], dim=1)
        return self.pred_alpha(z_compressed)


class AlphaProbe16v1(nn.Module):

    def __init__(self, args, additional_dims=0):
        super().__init__()
        self.use_batchnorm = False if "use_batchnorm_probe" not in args.keys() else args['use_batchnorm_probe']
        self.compress_z = [nn.Conv2d(args['latent'] * 2, 1, 1, padding=0), nn.LeakyReLU()]
        if self.use_batchnorm:
            self.compress_z.extend([nn.BatchNorm2d(1)])
        self.compress_z.extend([nn.Flatten()])
        self.compress_z = nn.Sequential(*self.compress_z)
        flattend_dim = args['latent_width'] ** 2
        self.pred_alpha = nn.Sequential(*[nn.Linear(flattend_dim + additional_dims, 512), nn.ReLU(),
                                          nn.Linear(512, args['latent'] * 2, bias=False)])

    def forward(self, z, add_features):
        z_compressed = self.compress_z(z)
        z_compressed = torch.cat([z_compressed, add_features], dim=1)
        return self.pred_alpha(z_compressed)


class AlphaProbe16v2(nn.Module):

    def __init__(self, args, additional_dims=0):
        super().__init__()
        self.use_batchnorm = False if "use_batchnorm_probe" not in args.keys() else args['use_batchnorm_probe']
        self.compress_z = [nn.Conv2d(args['latent'] * 2, 1, 1, padding=0), nn.LeakyReLU()]
        if self.use_batchnorm:
            self.compress_z.extend([nn.BatchNorm2d(1)])
        self.compress_z.extend([nn.Flatten()])
        self.compress_z = nn.Sequential(*self.compress_z)
        flattend_dim = args['latent_width'] ** 2
        self.pred_alpha = nn.Sequential(*[nn.Linear(flattend_dim + additional_dims, 512), nn.ReLU(),
                                          nn.Linear(512, 1024), nn.ReLU(),
                                          nn.Linear(1024, args['latent'] * 2, bias=False)])

    def forward(self, z, add_features):
        z_compressed = self.compress_z(z)
        z_compressed = torch.cat([z_compressed, add_features], dim=1)
        return self.pred_alpha(z_compressed)


class AlphaProbe16Convex(nn.Module):

    def __init__(self, args, additional_dims=0):
        super().__init__()
        self.use_batchnorm = False if "use_batchnorm_probe" not in args.keys() else args['use_batchnorm_probe']
        self.compress_z = [nn.Conv2d(args['latent'] * 2, 1, 1, padding=0), nn.LeakyReLU()]
        if self.use_batchnorm:
            self.compress_z.extend([nn.BatchNorm2d(1)])
        self.compress_z.extend([nn.Flatten()])
        self.compress_z = nn.Sequential(*self.compress_z)
        flattend_dim = args['latent_width'] ** 2
        self.pred_alpha = nn.Sequential(*[nn.Linear(flattend_dim + additional_dims, 512), nn.ReLU(),
                                          nn.Linear(512, 1024), nn.ReLU(),
                                          nn.Linear(1024, args['latent'], bias=False)])

    def forward(self, z, add_features):
        z_compressed = self.compress_z(z)
        z_compressed = torch.cat([z_compressed, add_features], dim=1)
        return self.pred_alpha(z_compressed)


class AlphaProbe256v1(nn.Module):

    def __init__(self, args, additional_dims=0, reduction=32):
        """
                Outputs args['latent_width'] ** 2 coefficients
        Args:
            args:
            additional_dims:
            reduction:
        """
        super().__init__()
        compressed_channels = args['latent_width'] // 2
        self.use_batchnorm = False if "use_batchnorm_probe" not in args.keys() else args['use_batchnorm_probe']
        self.compress_z = [nn.Conv2d(args['latent'] * 2, compressed_channels, 1, padding=0), nn.LeakyReLU()]
        if self.use_batchnorm:
            self.compress_z.extend([nn.BatchNorm2d(compressed_channels)])
        self.compress_z.extend([nn.Flatten()])
        self.compress_z = nn.Sequential(*self.compress_z)
        channels = (args['latent_width'] ** 2 * compressed_channels) + additional_dims
        channels_out = args['latent_width'] ** 2
        self.pred_alpha = nn.Sequential(*[nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(),
                                          nn.Linear(channels // reduction, channels, bias=False), nn.ReLU(),
                                          nn.Linear(channels, channels_out, bias=False)])

    def forward(self, z, add_features):
        z_compressed = self.compress_z(z)
        z_compressed = torch.cat([z_compressed, add_features], dim=1)
        return self.pred_alpha(z_compressed)


class AlphaProbe16ExBN(nn.Module):

    def __init__(self, args, additional_dims=0):
        super().__init__()
        self.use_batchnorm = False if "use_batchnorm_probe" not in args.keys() else args['use_batchnorm_probe']
        self.compress_z = []
        if self.use_batchnorm:
            self.compress_z.extend([nn.BatchNorm2d(args['latent'] * 2)])
        self.compress_z = [nn.Conv2d(args['latent'] * 2, 1, 1, padding=0), nn.LeakyReLU()]
        if self.use_batchnorm:
            self.compress_z.extend([nn.BatchNorm2d(1)])
        self.compress_z.extend([nn.Flatten()])
        self.compress_z = nn.Sequential(*self.compress_z)
        flattend_dim = args['latent_width'] ** 2
        self.pred_alpha = nn.Sequential(*[nn.Linear(flattend_dim + additional_dims, 512),
                                          nn.Linear(512, args['latent'] * 2, bias=False)])

    def forward(self, z, add_features):
        z_compressed = self.compress_z(z)
        z_compressed = torch.cat([z_compressed, add_features], dim=1)
        return self.pred_alpha(z_compressed)


class AlphaDecoder(nn.Module):

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

        self.dec = Decoder(scales, args['depth'], args['latent'], args['colors'],
                           n_res_block=args['n_res_block'], use_batchnorm=args['use_batchnorm'],
                           use_sigmoid=args['use_sigmoid']).to(args['device'])

    def forward(self, z):
        return self.dec(z)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

