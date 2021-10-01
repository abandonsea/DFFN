#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:29 2021

@author: Pedro Vieira
@description: Implements the train function for the DFFN network published in https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm

from tools import *
from net import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset settings
DATASET = 'PaviaU'  # PaviaU; KSC; Salinas
FOLDER = './Datasets/'  # Dataset folder
SAMPLE_SIZE = 23  # Hyper parameter: patch size
SAMPLE_BANDS = 5  # Number of bands after applying PCA
GENERATE_SAMPLE = False  # whether randomly generated training samples are ready
MAX_SAMPLES_PER_CLASS = None  # max training samples per class (use None for no limit)
# BATCH_SIZE_PER_CLASS = SAMPLES_PER_CLASS // 2  # batch size of each class
TRAIN_SPLIT = 0.7
FLIP_ARGUMENT = False  # Whether use argumentation with flipping data; default: False
ROTATED_ARGUMENT = False  # Whether use argumentation with rotated data; default: False

# Hyper parameters
NUM_RUNS = 1  # The amount of time the whole experiment should run
NUM_EPOCHS = 5  # Number of epochs per run
TEST_NUM = 0  # The total number of tests during the training process
LEARNING_RATE = 0.1
MOMENTUM = 0.9


# Train
def train():
    # Load raw dataset, apply PCA and normalize dataset.
    data = HSIData(DATASET, FOLDER, SAMPLE_BANDS)

    # Run training
    for run in range(NUM_RUNS):
        print("Running an experiment with run {}/{}".format(run + 1, NUM_RUNS))

        # Generate samples or read existing samples
        if GENERATE_SAMPLE:
            train_gt, test_gt = data.split_ground_truth(TRAIN_SPLIT, MAX_SAMPLES_PER_CLASS)
            data.save_samples(train_gt, test_gt, MAX_SAMPLES_PER_CLASS, run)
        else:
            train_gt, test_gt = data.load_samples(MAX_SAMPLES_PER_CLASS, run)

        # Create train and test dataset objects
        train_dataset = HSIDataset(data.image, train_gt, SAMPLE_SIZE, data_augmentation=True)
        test_dataset = HSIDataset(data.image, test_gt, SAMPLE_SIZE, data_augmentation=False)

        # Create train and test loaders
        train_loader = DataLoader(train_dataset)
        test_loader = DataLoader(test_dataset)

        # Setup model, optimizer and loss
        model = DFFN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

        total_steps = len(train_loader)
        for epoch in range(NUM_EPOCHS):
            print("Running epoch {}/{}".format(epoch + 1, NUM_EPOCHS))
            for i, (images, labels) in enumerate(train_loader):
                # image should have size 23x23x5
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 1000 == 0:
                    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}')

        print("Finished training!")


# Main function
def main():
    train()


if __name__ == main:
    main()