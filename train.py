#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:29 2021

@author: Pedro Vieira
@description: Implements the train function for the DFFN network published in https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018
"""

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from tqdm import tqdm

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
TRAIN_SPLIT = 0.8  # Fraction from the dataset used for training
TRAIN_BATCH_SIZE = 100  # Batch size for every train iteration
TEST_BATCH_SIZE = 20  # Batch size for every test iteration
SAMPLE_SIZE = 23  # Hyper parameter: patch size
SAMPLE_BANDS = 5  # Number of bands after applying PCA
GENERATE_SAMPLE = True  # Whether the samples should be generated (False to load previously saved samples)
MAX_SAMPLES_PER_CLASS = None  # max training samples per class (use None for no limit)

# Hyper parameters
NUM_RUNS = 1  # The amount of time the whole experiment should run
NUM_EPOCHS = 6  # Number of epochs per run
LEARNING_RATE = 0.1  # Initial learning rate
MOMENTUM = 0.9  # Momentum of optimizer
GAMMA = 0.1  # Gamma parameter for the lr scheduler
SCHEDULER_STEP = 3  # Step size for the lr scheduler

# Other options
PRINT_FREQUENCY = 50  # The amount of iterations between every step/loss print
TEST_NETWORK = False  # Whether to test network immediately after training a run


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
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # Setup model, optimizer and loss
        model = DFFN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        # Scheduler
        step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=SCHEDULER_STEP,
                                                            gamma=GAMMA)

        # Run epochs
        running_loss = 0.0
        running_correct = 0
        total_steps = len(train_loader)
        test_size = len(test_loader)
        print('Train size: ', total_steps)
        print('Test size: ', test_size)
        for epoch in range(NUM_EPOCHS):
            print("RUNNING EPOCH {}/{}".format(epoch + 1, NUM_EPOCHS))

            # Run iterations
            for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
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

                # Compute intermediate results for visualization
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                running_correct += (predicted == labels).sum().item()

                # Print steps and loss every PRINT_FREQUENCY
                if (i + 1) % PRINT_FREQUENCY == 0:
                    tqdm.write(f'\tEpoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{total_steps}]\tLoss: {loss.item():.4f}')
                    writer.add_scalar('training loss', running_loss / PRINT_FREQUENCY, epoch * total_steps + i)
                    writer.add_scalar('accuracy', running_correct / PRINT_FREQUENCY, epoch * total_steps + i)
                    running_loss = 0.0
                    running_correct = 0

        print("Finished training!")

        # Run testing if selected
        if TEST_NETWORK:
            labels_pr = []
            prediction_pr = []
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for images, labels in test_loader:
                    images = images.reshape(-1, 28 * 28).to(device)
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

                # Accuracy per class
                classes = range(10)
                for i in classes:
                    labels_i = labels_pr == i
                    prediction_i = prediction_pr[:, i]
                    writer.add_pr_curve(str(i), labels_i, prediction_i, global_step=0)


# Main function
def main():
    train()
    writer.close()


if __name__ == '__main__':
    main()
