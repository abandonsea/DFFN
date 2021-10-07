#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:29 2021

@author: Pedro Vieira
@description: Implements util functions to be used during train and/or test
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import random
import os


# Dataset class
class HSIData:
    """Stores dataset raw image and labels and applies pre-processing"""

    def __init__(self, dataset_name, target_folder='./datasets/', num_bands=5):
        self.dataset_name = dataset_name
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

        img = np.asarray(img, dtype='float32')
        self.image, self.pca, _, _ = self.apply_dimension_reduction(img, num_bands)

    @staticmethod
    def apply_dimension_reduction(image, num_bands=5):
        assert(num_bands < image.shape[2], 'The amount of bands should be smaller than the number image channels')
        image_height, image_width, image_bands = image.shape
        flat_image = np.reshape(image, (image_height * image_width, image_bands))

        # Normalize data before applying PCA. Range [-1, 1]
        sca1 = StandardScaler()
        sca1.fit(flat_image)
        norm1_img = sca1.transform(flat_image)

        # Apply PCA to reduce the number of bands to num_bands
        pca = PCA(int(num_bands))
        pca.fit(norm1_img)
        pca_img = pca.transform(norm1_img)

        # Normalize data after applying PCA. Range [-1, 1] (Is it really necessary?)
        sca2 = StandardScaler()
        sca2.fit(pca_img)
        norm2_img = sca2.transform(pca_img)

        out_img = np.reshape(norm2_img, (image_height, image_width, num_bands))

        return out_img, pca, sca1, sca2  # Returning transformers for future usage

    # Split ground-truth pixels into train, test, val
    def sample_dataset(self, train_size=0.8, val_size=0.1, max_train_samples=None):
        assert(train_size >= 1 and train_size > 0, 'Train set size should be a value between 0 and 1')
        assert(val_size > 1 and val_size >= 0, 'Validation set size should be a value between 0 and 1')
        assert (train_size + val_size < 1, 'Train and validation sets should not use the whole dataset')

        # Get train samples and non-train samples (== test samples, when there is no validation set)
        train_gt, test_gt = self.split_ground_truth(self.ground_truth, train_size, max_train_samples)

        val_gt = None
        if val_size > 0:
            relative_val_size = val_size / (1 - train_size)
            val_gt, test_gt = self.split_ground_truth(test_gt, relative_val_size)

        return train_gt, test_gt, val_gt

    @staticmethod
    def split_ground_truth(ground_truth, set1_size, max_samples=None):
        set1_gt = np.zeros_like(ground_truth)
        set2_gt = np.copy(ground_truth)

        set1_index_list = []
        for c in np.unique(ground_truth):
            if c == 0:
                continue
            class_indices = np.nonzero(ground_truth == c)
            index_tuples = list(zip(*class_indices))  # Tuples with (x, y) index values

            num_samples_set1 = int(np.ceil(set1_size * len(index_tuples)))
            set1_len = min(filter(lambda s: s is not None, [max_samples, num_samples_set1]))
            set1_index_list += random.sample(index_tuples, set1_len)

        set1_indices = tuple(zip(*set1_index_list))
        set1_gt[set1_indices] = ground_truth[set1_indices]
        set2_gt[set1_indices] = 0
        return set1_gt, set2_gt

    # Load samples from hard drive for every run.
    @staticmethod
    def load_samples(dataset_name, train_split, val_split, run):
        train_size = 'train_' + str(int(100 * train_split)) + '_'
        val_size = 'val_' + str(int(100 * val_split)) + '_'
        filename = train_size + val_size + 'run_ ' + str(run) + '.mat'
        sample_file = './dataset_split/' + dataset_name + '/' + filename
        data = io.loadmat(sample_file)
        train_gt = data['train_gt']
        test_gt = data['test_gt']
        val_gt = data['val_gt']
        return train_gt, test_gt, val_gt

    # Save samples for every run.
    @staticmethod
    def save_samples(dataset_name, train_gt, test_gt, val_gt, train_split, val_split, run):
        train_size = 'train_' + str(int(100 * train_split)) + '_'
        val_size = 'val_' + str(int(100 * val_split)) + '_'
        sample_dir = './dataset_split/' + dataset_name + '/'
        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)
        sample_file = sample_dir + train_size + val_size + '_run' + str(run) + '.mat'
        io.savemat(sample_file, {'train_gt': train_gt, 'test_gt': test_gt, 'val_gt': val_gt})

    # Load necessary test values
    # TODO: Implement this
    @staticmethod
    def load_environment(name, folder='./Datasets/', dataset='PaviaU'):
        # Dummies
        return 2, 4


# Dataset class based on PyTorch's
class HSIDataset(Dataset):
    """Dataset class based on PyTorch's"""

    def __init__(self, data, gt, sample_size=23, data_augmentation=True):
        super(HSIDataset, self).__init__()
        self.sample_size = sample_size
        self.data_augmentation = data_augmentation

        # Add padding according to sample_size
        pad_size = sample_size // 2
        self.data = np.pad(data, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='symmetric')
        self.label = np.pad(gt, pad_size, mode='constant')

        # Get indices to data based on ground-truth
        indices = []
        for c in np.unique(self.label):
            if c == 0:
                continue
            class_indices = np.nonzero(self.label == c)
            index_tuples = list(zip(*class_indices))
            indices += index_tuples

        # Shuffle indices to break label order
        self.indices = shuffle(indices, random_state=0)

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
