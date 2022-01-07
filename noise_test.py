#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:33 2021

@author: Pedro Vieira
@description: Implements the test function for the SAE-3DDRN network adding noise to the input image
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import net.dffn as dffn_pavia
import net.dffn_salinas as dffn_salinas
import net.dffn_indian_pines as dffn_indian
from test import test_model
from utils.config import DFFNConfig
from utils.dataset import DFFNDataset
from utils.noise import add_noise
from utils.tools import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################
# SET TEST CONFIG FILE #
########################
PATH = 'experiments/'
EXPERIMENTS = ['server_03', 'server_salinas_01', 'server_indian_pines_01',
               'reduced_01_server_paviau_01', 'reduced_01_server_salinas_01', 'reduced_01_server_indian_pines_01',
               'reduced_05_server_paviau_01', 'reduced_05_server_salinas_01', 'reduced_05_server_indian_pines_02',
               'reduced_10_server_paviau_01', 'reduced_10_server_salinas_01', 'reduced_10_server_indian_pines_01']


NOISES = [['salt_and_pepper', 0], ['salt_and_pepper', 0.001], ['salt_and_pepper', 0.005], ['salt_and_pepper', 0.01],
          ['salt_and_pepper', 0.05],
          ['additive_gaussian', 0.05], ['additive_gaussian', 0.1], ['additive_gaussian', 0.3],
          ['additive_gaussian', 0.5], ['additive_gaussian', 1.0],
          ['multiplicative_gaussian', 0.1], ['multiplicative_gaussian', 0.3], ['multiplicative_gaussian', 0.5],
          ['multiplicative_gaussian', 1.0],
          ['section_mul_gaussian', 1], ['section_mul_gaussian', 2], ['section_mul_gaussian', 3],
          ['section_mul_gaussian', 4], ['section_mul_gaussian', 5], ['section_mul_gaussian', 6],
          ['section_mul_gaussian', 7], ['section_mul_gaussian', 8],
          ['single_section_gaussian', 1], ['single_section_gaussian', 2], ['single_section_gaussian', 3],
          ['single_section_gaussian', 4], ['single_section_gaussian', 5], ['single_section_gaussian', 6],
          ['single_section_gaussian', 7], ['single_section_gaussian', 8]]


def check_pavia_model(model_dict):
    has_block15 = False
    for k in model_dict:
        if 'block15' in k:
            has_block15 = True

    return has_block15


# Test SAE-3DDRN runs
def test(config_file):
    # Load config data from training
    cfg = DFFNConfig(config_file, test=True)

    # Set string modifier if testing best models
    test_best = 'best_' if cfg.test_best_models else ''
    if cfg.test_best_models:
        print('Testing best models from each run!')

    # Load raw dataset, apply PCA and normalize dataset.
    data = HSIData(cfg.dataset, cfg.data_folder, cfg.sample_bands)

    for run in range(cfg.num_runs):
        print(f'TESTING RUN {run + 1}/{cfg.num_runs}')

        _, test_gt, _ = HSIData.load_samples(cfg.split_folder, cfg.train_split, cfg.val_split, run)
        num_classes = len(np.unique(test_gt)) - 1  # Remove one for the "undefined" class

        for noise in NOISES:
            print(f'Using {noise[0]} noise with parameter: {noise[1]}')
            noisy_data = add_noise(data.raw_image, noise)
            pca_noisy_data = data.apply_transforms(noisy_data)

            # Load test ground truth and initialize test loader
            test_dataset = DFFNDataset(pca_noisy_data, test_gt, cfg.sample_size, data_augmentation=False)
            test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

            # Load model
            model_file = f'{cfg.exec_folder}runs/dffn_{test_best}model_run_{run}.pth'
            model_dict = torch.load(model_file, map_location=device)

            # Check whether the paviau architecture was used for this model
            is_pavia = check_pavia_model(model_dict)

            if is_pavia:
                model = nn.DataParallel(dffn_pavia.DFFN(cfg.sample_bands, num_classes))
            elif cfg.dataset == 'Salinas':
                model = nn.DataParallel(dffn_salinas.DFFN(cfg.sample_bands, num_classes))
            else:  # indian pines
                model = nn.DataParallel(dffn_indian.DFFN(cfg.sample_bands, num_classes))

            model.load_state_dict(model_dict)
            model.eval()

            # Set model to device
            model = model.to(device)

            # Test model from the current run
            report = test_model(model, test_loader)
            save_noise_results(cfg.results_folder, noise, report)


# Main for running test independently
def main():
    # Load experiments
    for experiment in EXPERIMENTS:
        config_file = PATH + experiment + '/config.yaml'

        test(config_file)


if __name__ == '__main__':
    main()
