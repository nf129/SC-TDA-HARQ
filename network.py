import numpy as np
from encoder_tda import *
from decoder import *
from encoder import *
from distortion import Distortion
from channel import Channel
from random import choice
import torch.nn as nn
import torch
# import tda_VR
import torchvision.transforms as transforms
# import pandas as pd
from tqdm import tqdm

class SC_TDA(nn.Module):
    def __init__(self, args, config):
        super(SC_TDA, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        encoder_tda_kwargs = config.encoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        self.encoder_sim28 = create_encoder_tda(**encoder_tda_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.downsample = config.downsample
        self.model = args.model

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, input_image_tda, given_SNR = None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        feature = self.encoder(input_image, input_image_tda, chan_param)

        CBR = feature.numel() / 2 / input_image.numel()
        # Feature pass channel
        if self.pass_channel:
            noisy_feature1 = self.feature_pass_channel(feature, chan_param)
            noisy_feature2 = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature1 = feature

        zero = torch.zeros_like(noisy_feature1)
        noisy_feature = torch.dstack((noisy_feature1, zero))
        # noisy_feature = torch.dstack((noisy_feature1, noisy_feature2))
        recon_image, image_noisy_tda = self.decoder(noisy_feature, chan_param)

        # HARQ
        corr = self.encoder_sim28(recon_image, image_noisy_tda)

        corr = torch.mean(corr, dim=1)
        corr = torch.mean(corr, dim=0)

        if corr < 0.5:
            noisy_feature = torch.dstack((noisy_feature1, noisy_feature2))
            recon_image, image_noisy_sim28 = self.decoder(noisy_feature, chan_param)

        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, image_noisy_tda, CBR, chan_param, mse.mean(), loss_G.mean(), corr
