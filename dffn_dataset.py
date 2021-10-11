#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 8 17:47 2021

@author: Pedro Vieira
@description: Implements dataset function to provide the data during training
"""

import torch
from torch.utils.data import Dataset
import numpy as np


# Dataset class based on PyTorch's
class DFFNDataset(Dataset):
    """Dataset class based on PyTorch's"""

    def __init__(self, data, gt, sample_size=23, data_augmentation=True):
        super(DFFNDataset, self).__init__()
        self.sample_size = sample_size
        self.data_augmentation = data_augmentation

        # Add padding according to sample_size
        pad_size = sample_size // 2
        self.data = np.pad(data, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='symmetric')
        self.label = np.pad(gt, pad_size, mode='constant')

        # Get indices to data based on ground-truth
        self.indices = []
        for c in np.unique(self.label):
            if c == 0:
                continue
            class_indices = np.nonzero(self.label == c)
            index_tuples = list(zip(*class_indices))
            self.indices += index_tuples

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        pad_size = self.sample_size // 2
        x1, y1, x2, y2 = x - pad_size, y - pad_size, x + pad_size + 1, y + pad_size + 1
        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y] - 1  # Subtract one to ignore label 0 (unlabeled)

        if self.data_augmentation:
            # Perform data augmentation
            data = self.apply_data_augmentation(data)

        # Transpose data for it to match the expected torch dimensions
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        return data, label

    @staticmethod
    def apply_data_augmentation(data):
        # Probability of flipping data
        p1 = np.random.random()
        if p1 < 0.334:
            data = np.fliplr(data)
        elif p1 < 0.667:
            data = np.flipud(data)

        # Probability of rotating image
        p2 = np.random.random()
        if p2 < 0.25:
            data = np.rot90(data)
        elif p2 < 0.5:
            data = np.rot90(data, 2)
        elif p2 < 0.75:
            data = np.rot90(data, 3)

        return data
