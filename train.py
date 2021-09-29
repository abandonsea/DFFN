#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:29 2021

@author: Pedro Vieira
@description: Implements the train function for the DFFN network published in https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018
"""

import torch
import torch.utils.data as torch_data
import numpy as np
import tqdm

from tools import *
from net import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Datase settings
DATASET = 'PaviaU'  # PaviaU; KSC; Salinas
SAMPLE_ALREADY = False  # whether randomly generated training samples are ready
SAMPLE_SIZE = 10  # training samples per class
BATCH_SIZE_PER_CLASS = SAMPLE_SIZE // 2  # batch size of each class
PATCH_SIZE = 5  # Hyper parameter: patch size
FLIP_ARGUMENT = False  # whether need data argumentation of flipping data; default: False
ROTATED_ARGUMENT = False  # whether need data argumentation of rotated data; default: False
SAMPLING_MODE = 'fixed_withone'  # fixed number for each class
FOLDER = './Datasets/'  # the dataset folder

# Hyper parameters
NUM_RUNS = 1  # the running times of the experiments
NUM_EPOCHS = 5
TEST_NUM = 0  # the total number of test in the training process
LEARNING_RATE = 0.1  # 0.01 good / 0.1 fast for SGD; 0.001 for Adam
MOMENTUM = 0.9


# Train
def train():
    # Load dataset
    img, gt, label_values, ignored_labels, rgb_bands = load_dataset(DATASET, FOLDER)

    # Run training
    for run in range(NUM_RUNS):
        print("Running an experiment with run {}/{}".format(run + 1, NUM_RUNS))

        train_loader = torch_data.DataLoader()
        test_loader = torch_data.DataLoader()

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