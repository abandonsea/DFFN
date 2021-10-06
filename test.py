#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 5 17:57 2021

@author: Pedro Vieira
@description: Implements the test function for the DFFN network published in https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018
"""

import torch
import torch.nn.functional as f
import torch.utils.data as Torchdata
import numpy as np
from tqdm import tqdm

from tools import *
from net import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters setting
DATASET = 'PaviaU'  # PaviaU; Salinas; KSC
FOLDER = './Datasets/'  # the dataset folder
CHECKPOINT_FOLDER = 'checkpoints/' + DATASET + '/'
BATCH_SIZE = 20  # Batch size for every test iteration


# Test DFFN
def test(test_loader=None, model=None, batch_size=BATCH_SIZE):
    # Load data if none is provided
    if test_loader is None or model is None:
        test_loader, model = load_test_environment()

    # Begin testing
    labels_pr = []
    prediction_pr = []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
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
        print(f'accuracy = {acc}')


# Main for running test independently
def main():
    test()


if __name__ == '__main__':
    main()
