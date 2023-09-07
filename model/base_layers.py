import torch
from torch import nn


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, scale_factor=None, bias=True, Nonlinearity=True):
        super(UpsampleLayer, self).__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=self.scale_factor)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.nonlinearity = nn.Sequential(nn.BatchNorm2d(out_channels), nn.LeakyReLU()) if Nonlinearity else None
        
    def forward(self, x):
        if self.scale_factor:
            x = self.upsample(x)
        x = self.convolution(self.pad(x))
        if self.nonlinearity:
            x = self.nonlinearity(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, scale_factor=None, bias=True, Nonlinearity=True):
        super(DownsampleLayer, self).__init__()
        self.scale_factor = scale_factor
        self.downsample = nn.MaxPool2d(self.scale_factor)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.nonlinearity = nn.Sequential(nn.BatchNorm2d(out_channels), nn.LeakyReLU()) if Nonlinearity else None
    
    def forward(self, x):
        if self.scale_factor:
            x = self.downsample(x)
        x = self.convolution(self.pad(x))
        if self.nonlinearity:
            x = self.nonlinearity(x)
        return x


