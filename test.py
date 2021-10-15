#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 5 17:57 2021

@author: Pedro Vieira
@description: Implements the test function for the DFFN network published in https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
from tqdm import tqdm

from utils.config import DFFNConfig
from utils.dataset import DFFNDataset
from utils.tools import *
from net.dffn import DFFN

# Import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################
# SET TEST CONFIG FILE #
########################
CONFIG_FILE = ''  # Empty string to load default 'config.yaml'


# Test DFFN runs
def test():
    # Load config data from training
    config_file = 'config.yaml' if not CONFIG_FILE else CONFIG_FILE
    cfg = DFFNConfig(config_file, test=True)

    # Start tensorboard
    writer = None
    if cfg.use_tensorboard:
        writer = SummaryWriter(cfg.tensorboard_folder)

    # Load processed dataset
    data = torch.load(cfg.exec_folder + 'proc_data.pth')

    for run in range(cfg.num_runs):
        print(f'TESTING RUN {run + 1}/{cfg.num_runs}')

        # Load test ground truth and initialize test loader
        _, test_gt, _ = HSIData.load_samples(cfg.split_folder, cfg.train_split, cfg.val_split, run)
        test_dataset = DFFNDataset(data, test_gt, cfg.sample_size, data_augmentation=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

        # Load model
        model_file = cfg.exec_folder + 'dffn_model_run_' + str(run) + '.pth'
        model = DFFN()
        model.load_state_dict(torch.load(model_file))
        model.eval()

        # Set model to device
        if device == 'cuda':
            model = nn.DataParallel(model)
        model = model.to(device)

        # Test model from the current run
        report = test_model(model, test_loader, writer)
        filename = cfg.results_folder + 'test.txt'
        save_results(filename, report, run)

    if cfg.use_tensorboard:
        writer.close()


# Function for performing the tests for a given model and data loader
def test_model(model, loader, writer=None):
    labels_pr = []
    prediction_pr = []
    with torch.no_grad():
        total_predicted = np.array([], dtype=int)
        total_labels = np.array([], dtype=int)
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            # for images, labels in loader:
            # Get input and compute model output
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Get predicted outputs
            _, predicted = torch.max(outputs, 1)

            # Save total values for analysis
            total_predicted = np.append(total_predicted, predicted.cpu().numpy())
            total_labels = np.append(total_labels, labels.cpu().numpy())

        report = get_report(total_predicted, total_labels)
        print(f'- Overall accuracy: {report["overall_accuracy"]:f}')
        print(f'- Average accuracy: {report["average_accuracy"]:f}')
        print(f'- Kappa coefficient: {report["kappa"]:f}')

        if writer is not None:
            # Accuracy per class
            classes = range(9)
            for i in classes:
                labels_i = labels_pr == i
                prediction_i = prediction_pr[:, i]
                writer.add_pr_curve(str(i), labels_i, prediction_i, global_step=0)

    return report


# Compute OA, AA and kappa from the results
def get_report(y_pr, y_gt):
    classify_report = metrics.classification_report(y_gt, y_pr)
    confusion_matrix = metrics.confusion_matrix(y_gt, y_pr)
    class_accuracy = metrics.precision_score(y_gt, y_pr, average=None)
    overall_accuracy = metrics.accuracy_score(y_gt, y_pr)
    average_accuracy = np.mean(class_accuracy)
    kappa_coefficient = kappa(confusion_matrix, 5)

    # Save report values
    report = {
        'classify_report': classify_report,
        'confusion_matrix': confusion_matrix,
        'class_accuracy': class_accuracy,
        'overall_accuracy': overall_accuracy,
        'average_accuracy': average_accuracy,
        'kappa': kappa_coefficient
    }
    return report


# Compute kappa coefficient
def kappa(confusion_matrix, k):
    data_mat = np.mat(confusion_matrix)
    p_0 = 0.0
    for i in range(k):
        p_0 += data_mat[i, i] * 1.0
    x_sum = np.sum(data_mat, axis=1)
    y_sum = np.sum(data_mat, axis=0)
    p_e = float(y_sum * x_sum) / np.sum(data_mat)**2
    oa = float(p_0 / np.sum(data_mat) * 1.0)
    cohens_coefficient = float((oa - p_e) / (1 - p_e))
    return cohens_coefficient


# Main for running test independently
def main():
    test()


if __name__ == '__main__':
    main()
