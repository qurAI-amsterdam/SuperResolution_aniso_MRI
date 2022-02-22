import numpy as np
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


class MyView(nn.Module):
    def __init__(self, shape):
        super(MyView, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.contiguous().view(*self.shape)


class BottleNeck(nn.Module):

    def __init__(self, architecture):
        super(BottleNeck, self).__init__()
        self.block1 = nn.Sequential()
        self.block1.add_module("conv_to_latent_dim", nn.Conv2d(architecture.channel_list[-1], architecture.latent_dim_chan, 3,
                                                               stride=1, dilation=1, padding=1))

    def forward(self, x):
        return self.block1(x)


class BasicEncoderBlock(nn.Module):

    def __init__(self, channels_in, channels_out, kernel=3, padding=1, downsample=True, use_batchnorm=False, dropout_perc=0.):
        super(BasicEncoderBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(channels_in, channels_in, kernel, stride=1, dilation=1, padding=padding)
        self.non_linear = nn.LeakyReLU()
        self.conv2d_2 = nn.Conv2d(channels_in, channels_out, kernel, stride=1, dilation=1, padding=padding)
        self.batchnorm = nn.BatchNorm2d(channels_out)
        self.max_pool = nn.AvgPool2d(2, stride=None, padding=0)
        self.downsample = downsample
        self.use_batchnorm = use_batchnorm
        self.dropout_perc = dropout_perc
        self.dropout = nn.Dropout2d(p=self.dropout_perc)

    def forward(self, x):
        x = self.non_linear(self.conv2d_1(x))
        x = self.non_linear(self.conv2d_2(x))
        if self.use_batchnorm:
            x = self.batchnorm(x)
        if self.dropout_perc != 0:
            x = self.dropout(x)
        if self.downsample:
            x = self.max_pool(x)
        return x


class BasicDecoderBlock(nn.Module):

    def __init__(self, channels_in, channels_out, kernel=3, padding=1, do_upsample=True, dropout_perc=0.):
        super(BasicDecoderBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(channels_in, channels_in, kernel, stride=1, dilation=1, padding=padding)
        self.non_linear = nn.LeakyReLU()
        self.conv2d_2 = nn.Conv2d(channels_in, channels_out, kernel, stride=1, dilation=1, padding=padding)
        self.batchnorm = nn.BatchNorm2d(channels_out)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.do_upsample = do_upsample
        self.dropout_perc = dropout_perc
        self.dropout = nn.Dropout2d(p=self.dropout_perc)

    def forward(self, x):
        x = self.non_linear(self.conv2d_1(x))
        x = self.non_linear(self.conv2d_2(x))
        if self.dropout_perc != 0:
            x = self.dropout(x)
        if self.do_upsample:
            x = self.upsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, block_list, channel_list, downsample_list, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        # Encoder ********************************************************
        self.encoder = nn.Sequential()
        for block_id, num_layers in enumerate(block_list):
            downsample = downsample_list[block_id]
            out_channels = channel_list[block_id]
            block = BasicEncoderBlock(in_channels, out_channels, kernel=3, dropout_perc=0., use_batchnorm=False, padding=1,
                                      downsample=downsample)
            self.encoder = nn.Sequential(self.encoder, block)
            in_channels = out_channels
        self.bottleneck = [nn.Conv2d(in_channels, in_channels, 3, stride=1, dilation=1, padding=1), nn.LeakyReLU(),
                           nn.Conv2d(in_channels, self.latent_dim, 3, stride=1, dilation=1, padding=1), nn.LeakyReLU()]
        self.bottleneck = nn.Sequential(*self.bottleneck)
        self.apply(weights_init)

    def forward(self, x):
        x = self.encoder(x)
        return self.bottleneck(x)

    # def _make_conv_block(self, block_id, num_layers, channels_in, channels_out, kernel=3, stride=1, dilation=1, p_dropout=0., padding=0,
    #                      batch_norm=False, non_linearity=nn.ReLU, downsample=False):
    #     block = nn.Sequential()
    #     for i in range(num_layers):
    #         l_id = str(block_id) + "_" + str(i)
    #         block.add_module("conv2d_" + l_id, nn.Conv2d(channels_in, channels_out, kernel, stride=stride, dilation=dilation,
    #                                                               padding=padding))
    #         block.add_module("non_linear_" + l_id, non_linearity())
    #         if batch_norm:
    #             block.add_module("batchnorm_" + l_id, nn.BatchNorm2d(channels_out))
    #         if p_dropout != 0:
    #             block.add_module("dropout_layer" + l_id, nn.Dropout2d(p=p_dropout))
    #         channels_in = channels_out
    #
    #     if downsample:
    #         block.add_module("maxPooling2D" + str(block_id), nn.MaxPool2d(2, stride=None, padding=0))
    #     return block


class Decoder(nn.Module):

    def __init__(self, block_list, channel_list, downsample_list, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        in_channels = latent_dim
        # Important: we change the number of layers in the last block to 1:
        block_list[-1] = 1
        reversed_channels = channel_list[::-1][1:] + [1]  # IMPORTANT: [1] one output channel!
        # From latent space dims to feature channel shape
        self.decoder = nn.Sequential()

        # Note: in_channels is still equal to last #channels from self.channels list
        for block_id, num_layers in enumerate(block_list):
            do_upsample = downsample_list[block_id]
            out_channels = reversed_channels[block_id]
            layer = BasicDecoderBlock(in_channels, out_channels, kernel=3, do_upsample=do_upsample)
            self.decoder = nn.Sequential(self.decoder, layer)
            in_channels = out_channels

        self.decoder_conv = nn.Conv2d(in_channels, 1, 3, stride=1, dilation=1, padding=1)

    def forward(self, x):
        return self.decoder_conv(self.decoder(x))

    # @staticmethod
    # def _make_upsampling_block(block_id, num_layers, channels_in, channels_out, kernel=3, padding=1, batch_norm=False,
    #                            non_linearity=nn.ReLU, do_upsample=False):
    #     layer = nn.Sequential()
    #     for i in range(num_layers):
    #         l_id = str(block_id) + "_" + str(i)
    #         # layer.add_module("reflect_pad_" + l_id, nn.ReflectionPad2d(padding))
    #         layer.add_module("conv2d_" + l_id, nn.Conv2d(channels_in, channels_out, kernel_size=kernel, padding=1))
    #         layer.add_module("non_linear_" + l_id, non_linearity())
    #         if batch_norm and channels_out != 1:
    #             layer.add_module("batchnorm_" + l_id, nn.BatchNorm2d(channels_out))
    #         channels_in = channels_out
    #     if do_upsample:
    #         layer.add_module("upsample_" + str(block_id), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
    #     return layer


class AE(nn.Module):

    def __init__(self, architecture):
        """

        :param architecture: object AEConfig (see definition below) with configuration details
        """
        super(AE, self).__init__()
        self.block_list = architecture.block_list
        self.channel_list = architecture.channel_list
        self.downsample_list = architecture.downsample_list
        self.latent_dim_chan = architecture.latent_dim_chan
        self.sigmoid = nn.Sigmoid()
        self.encoder = Encoder(architecture.channels_input, self.block_list, self.channel_list, self.downsample_list, self.latent_dim_chan)
        # Decoder ***********************************************************
        self.decoder = Decoder(self.block_list, self.channel_list, self.downsample_list, self.latent_dim_chan)
        print(architecture.input_size, architecture.latent_dim_wh, self.channel_list[-1], self.downsample_list, self.latent_dim_chan)
        print("INFO - AE - compression {} ---> {}".format(architecture.input_size ** 2,
                                                             (self.latent_dim_chan * architecture.latent_dim_wh)))
        self.apply(weights_init)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return {'reconstruction': x}


class AEAdv(object):

    def __init__(self, architecture):
        super().__init__(architecture)
        self.block_list = architecture.block_list
        self.channel_list = architecture.channel_list
        self.downsample_list = architecture.downsample_list
        self.sigmoid = nn.Sigmoid()
        self.encoder = Encoder(architecture.channels_input, self.block_list, self.channel_list, self.downsample_list)
        # Decoder ***********************************************************
        self.decoder = Decoder(self.block_list, self.channel_list, self.downsample_list)

        self.to_latent_space = nn.Sequential()
        self.to_latent_space.add_module("ld_conv1", nn.Conv2d(architecture.channel_list[-1], architecture.latent_dim_chan, 1,
                                         stride=1, dilation=1, padding=0))
        self.to_latent_space.add_module('ld_batchnorm1', nn.BatchNorm2d(architecture.latent_dim_chan))
        self.to_latent_space.add_module('ld_LeakyRelu1', nn.LeakyReLU())
        self.from_latent_space = nn.Sequential()
        self.from_latent_space.add_module('ld_conv2', nn.Conv2d(architecture.latent_dim_chan, architecture.channel_list[-1], 1,
                                           stride=1, dilation=1, padding=0))
        self.from_latent_space.add_module('batchnorm2', nn.BatchNorm2d(architecture.channel_list[-1]))
        self.from_latent_space.add_module('LeakyRelu2', nn.LeakyReLU())
        print("INFO - AEAdv - compression {} ---> {}".format(architecture.input_size ** 2,
                                                            (architecture.latent_dim_chan * architecture.latent_dim_wh)))

    def encode(self, x):
        return self.to_latent_space(self.encoder(x))

    def decode(self, z):
        x = self.decoder(self.from_latent_space(z))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return {'reconstruction': x}


class DiscriminatorSpatial(nn.Module):

    def __init__(self, architecture):
        super().__init__()
        self.block_list = architecture.block_list
        self.channel_list = architecture.channel_list
        self.downsample_list = architecture.downsample_list
        self.encoder = nn.Sequential()

        in_channels = architecture.channels_input
        for block_id, num_layers in enumerate(self.block_list):
            downsample = self.downsample_list[block_id]
            out_channels = self.channel_list[block_id]
            block = BasicEncoderBlock(in_channels, out_channels, kernel=3, dropout_perc=0., use_batchnorm=False, padding=1,
                                      downsample=downsample)
            self.encoder = nn.Sequential(self.encoder, block)
            in_channels = out_channels
        flattened_dim = int(in_channels * architecture.latent_dim_wh)
        print(architecture.latent_dim_wh, in_channels)
        self.disc = [nn.Conv2d(in_channels, in_channels, 1, stride=1, dilation=1, padding=0),
                     MyView([-1, flattened_dim]),
                     nn.Linear(flattened_dim, 1)]
        self.disc = nn.Sequential(*self.disc)

    def forward(self, x):
        x = self.encoder(x)
        # scalar output per batch item. We're trying to regress alpha value for interp
        x = torch.squeeze(self.disc(x))
        return x


if __name__ == "__main__":
    import torchsummary
    from networks.model_configs import AEConfig, AEAdvConfig

    input_size = 128
    batch_size = 64
    ae_config = AEConfig(input_size=input_size)
    model = AE(ae_config).to('cuda')
    img = torch.randn(batch_size, 1, input_size, input_size).to('cuda')
    img_x = model(img)
    # print(img.detach().cpu().numpy().shape)
    # torchsummary.summary(model, (1, input_size, input_size))

    aeadv_config = AEAdvConfig(input_size)
    model = DiscriminatorSpatial(aeadv_config).to('cuda')
    alpha = model(img)
    print("Alpha output ", alpha.detach().cpu().numpy().shape)
    torchsummary.summary(model, (1, input_size, input_size))

