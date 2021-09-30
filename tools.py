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
        ignored_labels = [0]
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
        ignored_labels.append(0)
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

    # Get train test split
    def split_ground_truth(self, train_size, mode='fixed_with_one'):
        # TODO: Optimized code and check for correctness
        indices = np.nonzero(self.ground_truth)
        x = list(zip(*indices))  # x,y features
        y = self.ground_truth[indices].ravel()  # classes
        train_gt = np.zeros_like(self.ground_truth)
        test_gt = np.zeros_like(self.ground_truth)
        if train_size > 1:
            train_size = int(train_size)
            if mode.find('random') != -1:
                train_size = float(train_size) / 100

        if mode == 'random_with_one':
            train_indices = []
            test_gt = np.copy(self.ground_truth)
            for c in np.unique(self.ground_truth):
                if c == 0:
                    continue
                indices = np.nonzero(self.ground_truth == c)
                x = list(zip(*indices))  # x,y features
                train_len = int(np.ceil(train_size * len(x)))
                train_indices += random.sample(x, train_len)
            index = tuple(zip(*train_indices))
            train_gt[index] = self.ground_truth[index]
            test_gt[index] = 0

        elif mode == 'fixed_with_one':
            train_indices = []
            test_gt = np.copy(self.ground_truth)
            for c in np.unique(self.ground_truth):
                if c == 0:
                    continue
                indices = np.nonzero(self.ground_truth == c)
                x = list(zip(*indices))  # x,y features

                train_indices += random.sample(x, train_size)
            index = tuple(zip(*train_indices))
            train_gt[index] = self.ground_truth[index]
            test_gt[index] = 0
        else:
            raise ValueError("{} sampling is not implemented yet.".format(mode))
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

# Adapts matlab code
# def generate_index(dataset, dataset_gt, no_classes):
#     per_class_num = []
#     if dataset == 'indian_pines':
#         per_class_num = [5, 143, 83, 24, 49, 73, 3, 48, 2, 98, 245, 60, 21, 126, 39, 10]
#         # For Indian Pines: 10 % sampling.
#     elif dataset == 'paviau':
#         per_class_num = [132, 372, 42, 62, 27, 101, 27, 74, 19]
#         # For University of Pavia: 2 % sampling.
#     elif dataset == 'salinas':
#         per_class_num = [11, 19, 10, 7, 14, 20, 18, 57, 32, 17, 6, 10, 5, 6, 37, 10]
#         # For Salinas: 0.5 % sampling.
#
#     Train_Label = []
#     Train_index = []
#     train_data = []
#     test_data = []
#     train_label = []
#     test_label = []
#     train_index = []
#     test_index = []
#     index_len = []
#
#     for class_id in range(1, no_classes + 1):
#         label_gt = np.array(dataset_gt)
#         index_ii = np.where(label_gt == class_id)
#         rand_order = range(index_ii.shape[0]) # TODO: Random permutation
#         class_ii = np.ones(index_ii.shape[0]) * class_id
#         Train_Label.append(class_ii)
#         Train_index.append(index_ii)
#
#         num_train = per_class_num[class_id]
#         # num_train = floor(length(index_ii) * percent)
#         train_ii = rand_order(:, 1: num_train)
#         train_index = [train_index index_ii(train_ii)]
#         # train_index = [train_index index_ii(train_ii)]
#
#         test_index_temp = index_ii
#         test_index_temp(:, train_ii)=[]
#         test_index = [test_index test_index_temp]
#         # test_index = [test_index test_index_temp]
#
#         train_label = [train_label class_ii(:, 1: num_train)]
#         test_label = [test_label class_ii(num_train + 1:end)]
#
#     unlabeled_index = find(label_gt == 0)
#     order = randperm(length(unlabeled_index))
#     unlabeled_index = unlabeled_index(order)
#     unlabeled_label = zeros(1, length(unlabeled_index))