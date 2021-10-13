#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:41 2021

@author: Pedro Vieira @description: Implements blocks for the network described in
https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018/blob/master/prototxt_files/train_paviau.prototxt
"""

import torch
import torch.nn as nn


# Network design
# Based on the DFFN implementation for the PaviaU dataset.
class ScaleLayer(nn.Module):
    """Based on the scale layer from the Caffe Framework"""

    def __init__(self, num_channels=64, bias_term=True):
        super(ScaleLayer, self).__init__()
        self.weights = torch.ones(num_channels, requires_grad=True)
        self.bias = None
        if bias_term:
            self.bias = torch.zeros(num_channels, requires_grad=True)

    def forward(self, x):
        assert (x.size(1) == self.weights.size(0))
        for channel in range(x.size(1)):
            x[:, channel, :, :] *= self.weights[channel].item()
            if self.bias is not None:
                x[:, channel, :, :] += self.bias[channel].item()
        return x


class ConvBlock(nn.Module):
    """Basic conv-block"""

    def __init__(self, input_channels, feature_dim=64, padding=1, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv_batch_norm = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(feature_dim, momentum=1, affine=True))
        self.scale = ScaleLayer(feature_dim)

    def forward(self, x):
        out = self.conv_batch_norm(x)
        out = self.scale(out)
        return out


class ResBlock(nn.Module):
    """"Res-block with two conv-blocks"""

    def __init__(self, input_channels, feature_dim=64, stride=1, final_relu=True, identity_transform=None):
        super(ResBlock, self).__init__()
        self.final_relu = final_relu
        self.relu = nn.ReLU()
        self.conv_block1 = ConvBlock(input_channels, feature_dim, stride=stride)
        self.conv_block2 = ConvBlock(feature_dim, feature_dim=feature_dim)
        self.identity_transform = identity_transform

    def forward(self, x):
        identity = x
        if self.identity_transform is not None:
            identity = self.identity_transform(x)

        out = self.relu(self.conv_block1(x))
        out = self.conv_block2(out)
        out += identity
        if self.final_relu:
            out = self.relu(out)
        return out
