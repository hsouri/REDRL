from .base_layers import DownsampleLayer as DS
from .base_layers import UpsampleLayer as US
from torchvision import models
from .pix2pixmodels import UnetGenerator
from .nrp import NRP, NRP_resG
import torch
from torch import nn
import pdb
from tools.path import Paths


class ResNet_AE(nn.Module):
    def __init__(self, cfg):
        super(ResNet_AE, self).__init__()
        self.cfg = cfg
        self.device = torch.device(self.cfg.Device)
        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.decoder = nn.Sequential(US(512, 512, 3, 1, 2), US(512, 256, 3, 1, 2), US(256, 128, 3, 1, 2), US(128, 64, 3, 1, 2), US(64, 64, 3, 1, 2) ,US(64, 3, 1, 1, Nonlinearity=False))
        self.mu = nn.Sequential(DS(512, 64, 3, 1))
        self.logvar = nn.Sequential(DS(512, 64, 3, 1))
        self.features = nn.Sequential(DS(64, 512, 3, 1))
        self.initialize()
        self.to(self.device)

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)

    def forward(self, x):
        feat = self.encoder(x)
        #mu, logvar = self.mu(feat), self.logvar(feat)
        #noise = torch.randn(mu.shape).to(self.device)
        #feat = self.features(mu + noise * torch.exp(0.5 * logvar)) if self.training else self.features(mu)
        recon = self.decoder(feat)
        #if self.training:
        #    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        #    return recon, KLD
        #else:
        return recon



class UpsampleConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, scale_factor=None, bias=True):
        super(UpsampleConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=self.scale_factor)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, filter):
        if self.scale_factor:
            filter = self.upsample(filter)
        filter = self.convolution(self.pad(filter))
        return filter


class DownsampleConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, scale_factor=None, bias=True):
        super(DownsampleConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.downsample = nn.MaxPool2d(self.scale_factor)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, filter):
        if self.scale_factor:
            filter = self.downsample(filter)
        filter = self.convolution(self.pad(filter))
        return filter


class SegNet_AE(nn.Module):
    def __init__(self, cfg):
        super(SegNet_AE, self).__init__()
        self.cfg = cfg
        self.device = torch.device(self.cfg.Device)
        self.Encoder = nn.Sequential(
            DownsampleConv2d(3, 64, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),
            DownsampleConv2d(64, 64, 3, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),
            DownsampleConv2d(64, 128, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),
            DownsampleConv2d(128, 128, 3, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),
            DownsampleConv2d(128, 256, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(),
            DownsampleConv2d(256, 256, 3, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(),
            DownsampleConv2d(256, 512, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(),
            DownsampleConv2d(512, 512, 1, 1,  bias=False), nn.BatchNorm2d(512), nn.LeakyReLU())
        ###########
        self.mu = nn.Sequential(DownsampleConv2d(512, 64, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU())
        self.logvar = nn.Sequential(DownsampleConv2d(512, 64, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU())
        self.featureVector = nn.Sequential(DownsampleConv2d(64, 512, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU())
        ###########
        self.Decoder = nn.Sequential(
            UpsampleConv2d(512, 512, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(),
            UpsampleConv2d(512, 256, 3, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(),
            UpsampleConv2d(256, 256, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(),
            UpsampleConv2d(256, 128, 3, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),
            UpsampleConv2d(128, 128, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),
            UpsampleConv2d(128, 64, 3, 1,  bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),
            UpsampleConv2d(64, 64, 3, 1, scale_factor=2, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),
            UpsampleConv2d(64, 3, 1, 1,  bias=False))

        self.initialize()
        #self.load_pretrained_weights()
        self.to(self.device)

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)

    
    def load_pretrained_weights(self):
        ckpt = torch.load(Paths.VAE_CKPT, map_location='cpu')['State_Dictionary']
        ckpt_copy = {}
        for key in ckpt.keys():
            ckpt_copy[key[7:]] = ckpt[key]
        self.load_state_dict(ckpt_copy)
        print('Variational AutoEncoder Reconstructor Network Pretrained weights loaded!')
        print('VAE Checkpoint is loaded!')


    def forward(self, x):
        feat = self.Encoder(x)
        #mu, logvar = self.mu(feat), self.logvar(feat)
        #noise = torch.randn(mu.shape).to(self.device)
        #feat = self.featureVector(mu + noise * torch.exp(0.5 * logvar)) if self.training else self.featureVector(mu)
        recon = self.Decoder(feat)
        #if self.training:
        #    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        #    return recon, KLD
        #else:
        return recon





         