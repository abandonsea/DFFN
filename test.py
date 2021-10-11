#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 5 17:57 2021

@author: Pedro Vieira
@description: Implements the test function for the DFFN network published in https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018
"""

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from config import DFFNConfig
from dffn_dataset import DFFNDataset
from tools import *
from net import DFFN

# Import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################
# SET TEST CONFIG FILE #
########################
CONFIG_FILE = './experiments/exec_01/config.yaml'


# Test DFFN runs
def test(writer=None):
    # Load config data from training
    cfg = DFFNConfig(CONFIG_FILE, test=True)

    # Load processed dataset
    data = torch.load(cfg.exec_folder + 'proc_data.pth')

    for run in range(cfg.num_runs):
        # Load test ground truth and initialize test loader
        _, test_gt, _ = HSIData.load_samples(cfg.split_folder, cfg.train_split, cfg.val_split, run)
        test_dataset = DFFNDataset(data, test_gt, cfg.sample_size, data_augmentation=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

        # Load model
        model_file = cfg.exec_folder + 'dffn_model_run_' + str(run) + '.pth'
        model = DFFN()
        model.load_state_dict(torch.load(model_file))
        model.eval()

        # Test model from the current run
        test_model(model, test_loader, writer)


# Function for performing the tests for a given model and data loader
def test_model(model, loader, writer=None):
    labels_pr = []
    prediction_pr = []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in loader:
            # Get input and compute model output
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predicted == labels).sum().item()

            class_predictions = [f.softmax(output, dim=0) for output in outputs]

            prediction_pr.append(class_predictions)
            labels_pr.append(predicted)

        prediction_pr = torch.cat([torch.stack(batch) for batch in prediction_pr])
        labels_pr = torch.cat(labels_pr)

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy = {acc}')

        # TODO: Also add measures like OA, AA and kappa
        if writer is not None:
            # Accuracy per class
            classes = range(10)
            for i in classes:
                labels_i = labels_pr == i
                prediction_i = prediction_pr[:, i]
                writer.add_pr_curve(str(i), labels_i, prediction_i, global_step=0)


# Main for running test independently
def main():
    writer = SummaryWriter('tensorboard')
    test()
    writer.close()


if __name__ == '__main__':
    main()
