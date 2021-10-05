#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:29 2021

@author: Pedro Vieira
@description: Implements the train function for the DFFN network published in https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools import *
from net import *

# Use tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/code_test')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset settings
DATASET = 'PaviaU'  # PaviaU; KSC; Salinas
FOLDER = './Datasets/'  # Dataset folder
TRAIN_SPLIT = 0.7  # Fraction from the dataset used for training
BATCH_SIZE = 10  # Batch size for every iteration
SAMPLE_SIZE = 23  # Hyper parameter: patch size
SAMPLE_BANDS = 5  # Number of bands after applying PCA
GENERATE_SAMPLE = True  # Whether the samples should be generated (False to load previously saved samples)
MAX_SAMPLES_PER_CLASS = 50  # max training samples per class (use None for no limit)

# Hyper parameters
NUM_RUNS = 1  # The amount of time the whole experiment should run
NUM_EPOCHS = 6  # Number of epochs per run
LEARNING_RATE = 0.1  # Initial learning rate
MOMENTUM = 0.9  # Momentum of optimizer
GAMMA = 0.1  # Gamma parameter for the lr scheduler
SCHEDULER_STEP = 3  # Step size for the lr scheduler
PRINT_FREQUENCY = 25  # The amount of iterations between every step/loss print


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
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Setup model, optimizer and loss
        model = DFFN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        # Scheduler
        step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=SCHEDULER_STEP,
                                                            gamma=GAMMA)

        # Run epochs
        total_steps = len(train_loader)
        for epoch in range(NUM_EPOCHS):
            print("RUNNING EPOCH {}/{}".format(epoch + 1, NUM_EPOCHS))

            # Run iterations
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

                # Print steps and loss every PRINT_FREQUENCY
                if (i + 1) % PRINT_FREQUENCY == 0:
                    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}')

        print("Finished training!")


# Main function
def main():
    train()


if __name__ == '__main__':
    main()
