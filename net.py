#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:59 2021

@author: Pedro Vieira
@description: Implements the network described in https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018/blob/master/prototxt_files/train_paviau.prototxt
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
        assert (x.size(1) == self.weights.size(1))
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

    def __init__(self, input_channels, feature_dim=64, stride=1, final_relu=True):
        super(ResBlock, self).__init__()
        self.final_relu = final_relu
        self.relu = nn.ReLU()
        self.conv_block1 = ConvBlock(input_channels, feature_dim, stride=stride)
        self.conv_block2 = ConvBlock(feature_dim, feature_dim=feature_dim)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv_block1(x))
        out = self.conv_block2(out)
        out += identity
        if self.final_relu:
            out = self.relu(out)
        return out


class DFFN(nn.Module):
    """DFFN architecture for PaviaU"""

    def __init__(self):
        super(DFFN, self).__init__()

        self.relu = nn.ReLU()

        # Stage 1
        self.pre_block1 = ConvBlock(5, feature_dim=16)
        self.block1 = ResBlock(16, feature_dim=16)
        self.block2 = ResBlock(16, feature_dim=16)
        self.block3 = ResBlock(16, feature_dim=16)
        self.block4 = ResBlock(16, feature_dim=16)
        self.block5 = ResBlock(16, feature_dim=16)

        # Stage 2
        self.dim_reduction1 = ConvBlock(16, feature_dim=32, padding=0, kernel_size=1, stride=2)
        self.block6 = ResBlock(16, feature_dim=32, stride=2)
        self.block7 = ResBlock(32, feature_dim=32)
        self.block8 = ResBlock(32, feature_dim=32)
        self.block9 = ResBlock(32, feature_dim=32)
        self.block10 = ResBlock(32, feature_dim=32)

        # Stage 3
        self.dim_reduction2 = ConvBlock(32, feature_dim=64, padding=0, kernel_size=1, stride=2)
        self.block11 = ResBlock(32, feature_dim=64, stride=2)
        self.block12 = ResBlock(64, feature_dim=64)
        self.block13 = ResBlock(64, feature_dim=64)
        self.block14 = ResBlock(64, feature_dim=64)
        self.block15 = ResBlock(64, feature_dim=64, final_relu=False)

        # Fuse stages
        self.dim_matching1 = ConvBlock(16, feature_dim=64, stride=4)
        self.dim_matching2 = ConvBlock(32, feature_dim=64, stride=2)
        self.pool = nn.AvgPool2d(1, 1)
        self.fc1 = nn.Linear(64, 9)

    def forward(self, x):
        #TODO: do it
        return x

# Initiate weights of net
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.05)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
    else:
        pass
