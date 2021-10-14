#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:56 2021

@author: Pedro Vieira @description: Implements the network described in
https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018/blob/master/prototxt_files/train_indian_pines.prototxt
"""

import torch
import torch.nn as nn

from net.blocks import ConvBlock, ResBlock


# Implementation of DFFN with the architecture used for Indian Pines.
# The sample size that was used is 25x25x3. The 5 channels are obtained by doing a PCA with the input data.
# The given input size should result in a 6x6x64 feature map when all stages are fused.
class DFFN(nn.Module):
    """DFFN architecture for Indian Pines"""

    def __init__(self):
        super(DFFN, self).__init__()

        self.relu = nn.ReLU()

        # Stage 1
        self.pre_block1 = ConvBlock(5, feature_dim=16)
        self.block1 = ResBlock(16, feature_dim=16)
        self.block2 = ResBlock(16, feature_dim=16)
        self.block3 = ResBlock(16, feature_dim=16)
        self.block4 = ResBlock(16, feature_dim=16)

        # Stage 2
        dim_reduction1 = ConvBlock(16, feature_dim=32, padding=0, kernel_size=1, stride=2)
        self.block5 = ResBlock(16, feature_dim=32, stride=2, identity_transform=dim_reduction1)
        self.block6 = ResBlock(32, feature_dim=32)
        self.block7 = ResBlock(32, feature_dim=32)
        self.block8 = ResBlock(32, feature_dim=32)

        # Stage 3
        dim_reduction2 = ConvBlock(32, feature_dim=64, padding=0, kernel_size=1, stride=2)
        self.block9 = ResBlock(32, feature_dim=64, stride=2, identity_transform=dim_reduction2)
        self.block10 = ResBlock(64, feature_dim=64)
        self.block11 = ResBlock(64, feature_dim=64)
        self.block12 = ResBlock(64, feature_dim=64, final_relu=False)

        # Fuse stages
        self.dim_matching1 = ConvBlock(16, feature_dim=64, stride=4)
        self.dim_matching2 = ConvBlock(32, feature_dim=64, stride=2)
        self.pool = nn.AvgPool2d(kernel_size=6)  # Input of this layer should have size 6x6x64
        self.fc1 = nn.Linear(64, 16)  # Indian Pines has 16 classes

    def forward(self, x):
        # Stage 1
        out = self.relu(self.pre_block1(x))
        out = self.block2(self.block1(out))
        out = self.block4(self.block3(out))
        stage1 = self.dim_matching1(out)

        # Stage 2
        out = self.block6(self.block5(out))
        out = self.block8(self.block7(out))
        stage2 = self.dim_matching2(out)

        # Stage 3
        out = self.block10(self.block9(out))
        out = self.block12(self.block11(out))

        # Fuse stages
        out += stage1 + stage2
        out = self.pool(out)
        out = out.view(-1, 64)
        out = self.fc1(out)
        # Softmax is done together with the Cross Entropy loss
        return out
