import torch.nn as nn
import torch
import math
from networks.acai_vanilla import Encoder, Decoder
# from utils.common import vae_loss, kl_divergence
# from torch.distributions.continuous_bernoulli import ContinuousBernoulli

"""
    
    pytorch reconstruction loss
    self.recon_loss = lambda x,y: F.binary_cross_entropy(x*0.5 + 0.5, y*0.5 + 0.5, size_average=False).\
                div(x.size(0))
     # Reconstruction loss.
        enc = self.generator.encode(x_batch)
        enc_sampled, enc_mu, enc_var, distn = self._sample_from_normal(enc)
        dec_enc = self.generator.decode(enc_sampled)
        # todo: move 0.5 into recon_loss
        recon_loss = self.recon_loss(dec_enc, x_batch)
             
     KL loss.
        prior = distns.normal.Normal(torch.zeros_like(enc_mu),
                                     torch.ones_like(enc_var))
        kl_loss = distns.kl.kl_divergence(distn, prior).mean()

        dec_enc_rand = self.generator.decode(prior.rsample())

        gen_loss = self.lamb*recon_loss + self.beta*kl_loss         
        
        self.lamb = 1, self.beta = 100 ?
    
    def _sample_from_normal(self, enc):
        enc_mu, enc_var = enc[:, 0:(enc.size(1)//2)], enc[:, (enc.size(1)//2)::]
        distn = distns.normal.Normal(enc_mu, enc_var)
        enc_sampled = distn.rsample()
        return enc_sampled, enc_mu, enc_var, distn
"""


class MyView(nn.Module):
    def __init__(self, shape):
        super(MyView, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.contiguous().view(*self.shape)


class VAE(nn.Module):

    def __init__(self, args):
        super().__init__()
        scales = int(round(math.log(args['width'] // args['latent_width'], 2)))
        self.use_multiple_gpu = True if ('gpu_ids' in args.keys() and len(args['gpu_ids']) > 1) else False
        if "n_res_block" not in args.keys():
            args['n_res_block'] = None
        if 'use_batchnorm' not in args.keys():
            args['use_batchnorm'] = False
        if 'use_sigmoid' not in args.keys():
            args['use_sigmoid'] = False
        if 'gpu_ids' not in args.keys():
            args['gpu_ids'] = [0]
        self.enc = Encoder(scales, args['depth'], args['latent'], args['colors'],
                           n_res_block=None, use_batchnorm=args['use_batchnorm']).to(args['device'])
        self.dec = Decoder(scales, args['depth'], args['latent'], args['colors'],
                           n_res_block=None, use_batchnorm=args['use_batchnorm'],
                           use_sigmoid=args['use_sigmoid']).to(args['device'])
        self.enc, self.dec = self.enc.to(args['device']), self.dec.to(args['device'])
        self.latent_flattend = args['latent'] * args['latent_width'] * args['latent_width']
        self.encoder_mu = [MyView([-1, self.latent_flattend]), nn.Linear(self.latent_flattend, self.latent_flattend)]
        self.encoder_logvar = [MyView([-1, self.latent_flattend]), nn.Linear(self.latent_flattend, self.latent_flattend)]
        self.encoder_mu = nn.Sequential(*self.encoder_mu)
        self.encoder_logvar = nn.Sequential(*self.encoder_logvar)
        # if self.use_multiple_gpu:
        #     print("Warning - VAE - INFO - moving linear layers to second GPU!")
        #     self.encoder_mu, self.encoder_logvar = self.encoder_mu.to('cuda:1'), self.encoder_logvar.to('cuda:1')
        # else:
        #     self.encoder_mu, self.encoder_logvar = self.encoder_mu.to('cuda:0'), self.encoder_logvar.to('cuda:0')
        self.unflatten = MyView([-1, args['latent'], args['latent_width'], args['latent_width']])

    def encode(self, x):
        x = self.enc(x)
        return x

    def decode(self, z):
        return self.dec(self.unflatten(z))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        z = self.encode(x)
        enc_logvar = self.encoder_logvar(z)
        if self.use_multiple_gpu:
            z = z.to('cuda:1')
        enc_mu = self.encoder_mu(z)
        if self.use_multiple_gpu:
            enc_mu = enc_mu.to('cuda:0')
        return self.decode(enc_mu)


class VAE2(nn.Module):

    def __init__(self, args):
        super().__init__()
        scales = int(round(math.log(args['width'] // args['latent_width'], 2)))
        self.use_multiple_gpu = True if ('gpu_ids' in args.keys() and len(args['gpu_ids']) > 1) else False
        if "n_res_block" not in args.keys():
            args['n_res_block'] = None
        if 'use_batchnorm' not in args.keys():
            args['use_batchnorm'] = False
        if 'use_sigmoid' not in args.keys():
            args['use_sigmoid'] = False
        if 'gpu_ids' not in args.keys():
            args['gpu_ids'] = [0]
        self.enc = Encoder(scales, args['depth'], args['latent'], args['colors'],
                           n_res_block=None, use_batchnorm=args['use_batchnorm']).to(args['device'])
        self.dec = Decoder(scales, args['depth'], args['latent'], args['colors'],
                           n_res_block=None, use_batchnorm=args['use_batchnorm'],
                           use_sigmoid=args['use_sigmoid']).to(args['device'])
        self.enc, self.dec = self.enc.to(args['device']), self.dec.to(args['device'])
        self.encode_flat_shape = args['latent'] * args['latent_width'] * args['latent_width']
        self.encoder_mu = [MyView([-1, self.encode_flat_shape]), nn.Linear(self.encode_flat_shape, args['latent'] )]
        self.encoder_logvar = [MyView([-1, self.encode_flat_shape]), nn.Linear(self.encode_flat_shape, args['latent'])]
        self.encoder_mu = nn.Sequential(*self.encoder_mu)
        self.encoder_logvar = nn.Sequential(*self.encoder_logvar)
        self.decoder_dense = nn.Sequential(*[nn.Linear(args['latent'], self.encode_flat_shape),
                                             MyView([-1, args['latent'], args['latent_width'], args['latent_width']])])
        # for vae2 this is a dummy thing that we call in trainer_vae. we do not need to reshape because
        # that happens in decoder_dense see above.
        self.unflatten = MyView([-1, args['latent']])

    def encode(self, x):
        x = self.enc(x)
        return x

    def decode(self, z):
        return self.dec(self.decoder_dense(z))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        z = self.encode(x)
        # enc_logvar = self.encoder_logvar(z)
        if self.use_multiple_gpu:
            z = z.to('cuda:1')
        enc_mu = self.encoder_mu(z)
        if self.use_multiple_gpu:
            enc_mu = enc_mu.to('cuda:0')
        return self.decode(enc_mu)
