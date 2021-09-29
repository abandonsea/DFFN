#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:29 2021

@author: Pedro Vieira
@description: Implements util functions to be used during train and/or test
"""

import torch
import torch.utils.data as Torchdata
import numpy as np
from scipy import io
import random
import os
from tqdm import tqdm


# Load data set
def load_dataset(dataset_name, target_folder='./Datasets/'):
    """Returns dataset raw image and labels"""

    folder = target_folder + dataset_name + '/'
    if dataset_name == 'IndianPines':
        # load the image
        img = io.loadmat(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']
        gt = io.loadmat(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        ignored_labels = [0]
    elif dataset_name == 'PaviaU':
        # load the image
        img = io.loadmat(folder + 'PaviaU.mat')['paviaU']
        gt = io.loadmat(folder + 'PaviaU_gt.mat')['paviaU_gt']
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        rgb_bands = (55, 41, 12)
        ignored_labels = [0]
    elif dataset_name == 'PaviaC':
        # Load the image
        img = io.loadmat(folder + 'Pavia.mat')['pavia']

        rgb_bands = (55, 41, 12)

        gt = io.loadmat(folder + 'Pavia_gt.mat')['pavia_gt']

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]

        ignored_labels = [0]
    elif dataset_name == 'Salinas':
        # Load the image
        img = io.loadmat(folder + 'Salinas_corrected.mat')['salinas_corrected']
        gt = io.loadmat(folder + 'Salinas_gt.mat')['salinas_gt']
        label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow',
                        'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained',
                        'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                        'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
        rgb_bands = (43, 21, 11)
        ignored_labels = [0]
    elif dataset_name == 'SalinaA':
        # Load the image
        img = io.loadmat(folder + 'SalinasA_corrected.mat')['salinasA_corrected']
        gt = io.loadmat(folder + 'SalinasA_gt.mat')['salinasA_gt']
        label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk',
                        'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk']
        rgb_bands = (43, 21, 11)
        ignored_labels = [0]
    elif dataset_name == 'KSC':
        # Load the image
        img = io.loadmat(folder + 'KSC.mat')['KSC']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = io.loadmat(folder + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]
    else:
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("Warning: NaN have been found in the data. "
              "It is preferable to remove them beforehand. "
              "Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    n_bands = img.shape[-1]
    for band in range(n_bands):
        min_val = np.min(img[:, :, band])
        max_val = np.max(img[:, :, band])
        img[:, :, band] = (img[:, :, band] - min_val) / (max_val - min_val)
    return img, gt, label_values, ignored_labels, rgb_bands


# Get train test split
def sample_gt(gt, train_size, mode='fixed_with_one'):
    indices = np.nonzero(gt)
    x = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)
        if mode == 'random':
            train_size = float(train_size) / 100  # dengbin:20181011

    if mode == 'random_with_one':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            x = list(zip(*indices))  # x,y features
            train_len = int(np.ceil(train_size * len(x)))
            train_indices += random.sample(x, train_len)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0

    elif mode == 'fixed_with_one':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            x = list(zip(*indices))  # x,y features

            train_indices += random.sample(x, train_size)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


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