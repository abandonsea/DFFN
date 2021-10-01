#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:29 2021

@author: Pedro Vieira
@description: Implements util functions to be used during train and/or test
"""

import torch
from sklearn.decomposition import PCA
import numpy as np
from scipy import io
import random
import os
from tqdm import tqdm


# Dataset class
class HSIDataset:
    def __init__(self, dataset_name, target_folder='./Datasets/'):
        """Returns dataset raw image and labels"""

        folder = target_folder + dataset_name + '/'
        self.rgb_bands = None
        self.label_values = None
        if dataset_name == 'IndianPines':
            img = io.loadmat(folder + 'Indian_pines_corrected.mat')['indian_pines_corrected']
            gt = io.loadmat(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
            self.label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                                 "Corn", "Grass-pasture", "Grass-trees",
                                 "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                                 "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                                 "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                                 "Stone-Steel-Towers"]
            self.rgb_bands = (43, 21, 11)  # AVIRIS sensor
        elif dataset_name == 'PaviaU':
            img = io.loadmat(folder + 'PaviaU.mat')['paviaU']
            gt = io.loadmat(folder + 'PaviaU_gt.mat')['paviaU_gt']
            self.label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                                 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                                 'Self-Blocking Bricks', 'Shadows']
            self.rgb_bands = (55, 41, 12)
        elif dataset_name == 'Salinas':
            img = io.loadmat(folder + 'Salinas_corrected.mat')['salinas_corrected']
            gt = io.loadmat(folder + 'Salinas_gt.mat')['salinas_gt']
            self.label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow',
                                 'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained',
                                 'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                                 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                                 'Vinyard_untrained', 'Vinyard_vertical_trellis']
            self.rgb_bands = (43, 21, 11)
        else:
            raise ValueError("{} dataset is unknown.".format(dataset_name))

        # Filter NaN values
        nan_mask = np.isnan(img.sum(axis=-1))
        if np.count_nonzero(nan_mask) > 0:
            print("Warning: NaN have been found in the data. "
                  "It is preferable to remove them beforehand. "
                  "Learning on NaN data is disabled.")
        img[nan_mask] = 0
        gt[nan_mask] = 0
        self.ground_truth = gt
        ignored_labels = [0]
        self.ignored_labels = list(set(ignored_labels))

        self.raw_image = np.asarray(img, dtype='float32')

    def apply_image_preprocessing(self, num_bands):
        # TODO: implement PCA (scikit-learn)
        num_bands = self.raw_image.shape[-1]

        # Normalize data
        img = self.raw_image
        for band in range(num_bands):
            min_val = np.min(img[:, :, band])
            max_val = np.max(img[:, :, band])
            img[:, :, band] = (img[:, :, band] - min_val) / (max_val - min_val)
        return img

    def num_classes(self):
        len(self.label_values) - len(self.ignored_labels)
        return

    # Split ground-truth pixels
    def split_ground_truth(self, train_size=0.75, max_train_samples=None):
        train_gt = np.zeros_like(self.ground_truth)
        test_gt = np.copy(self.ground_truth)

        # If train_size <= 1, use as a fraction of the dataset.
        # If train_size > 1, use as a percentage value and divide by 100.
        if train_size > 1:
            train_size = float(train_size) / 100

        train_index_list = []
        for c in np.unique(self.ground_truth):
            if c == 0:
                continue
            class_indices = np.nonzero(self.ground_truth == c)
            index_tuples = list(zip(*class_indices))  # Tuples with (x, y) index values

            num_train_samples = int(np.ceil(train_size * len(index_tuples)))
            train_len = min(filter(lambda s: s is not None, [max_train_samples, num_train_samples]))
            train_index_list += random.sample(index_tuples, train_len)

        train_indices = tuple(zip(*train_index_list))
        train_gt[train_indices] = self.ground_truth[train_indices]
        test_gt[train_indices] = 0

        return train_gt, test_gt

    def get_train_test_samples(self):
        # TODO: Get train and test samples for training (use matlab code as inspiration)
        return


# Get samples for every run
def get_sample(dataset_name, sample_size, run):
    sample_file = './TrainTestSplit/' + dataset_name + '/sample' + str(sample_size) + '_run' + str(run) + '.mat'
    data = io.loadmat(sample_file)
    train_gt = data['train_gt']
    test_gt = data['test_gt']
    return train_gt, test_gt


# Save samples for every run
def save_sample(train_gt, test_gt, dataset_name, sample_size, run):
    sample_dir = './TrainTestSplit/' + dataset_name + '/'
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    sample_file = sample_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    io.savemat(sample_file, {'train_gt': train_gt, 'test_gt': test_gt})
