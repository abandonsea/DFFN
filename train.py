#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:29 2021

@author: Pedro Vieira @description: Implements the train function for the DFFN network published in
https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DFFNConfig
from tools import *
from net import *
from test import test_model

# Import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Train
def train(writer=None):
    cfg = DFFNConfig('config.yaml')
    # Load raw dataset, apply PCA and normalize dataset.
    data = HSIData(cfg.dataset, cfg.data_folder, cfg.sample_bands)

    # Run training
    for run in range(cfg.num_runs):
        print("Running an experiment with run {}/{}".format(run + 1, cfg.num_runs))

        # Generate samples or read existing samples
        if cfg.generate_samples:
            train_gt, test_gt, val_gt = data.sample_dataset(cfg.train_split, cfg.val_split, cfg.max_samples)
            HSIData.save_samples(train_gt, test_gt, val_gt, cfg.split_folder, cfg.val_split, cfg.val_split, run)
        else:
            train_gt, test_gt, val_gt = HSIData.load_samples(cfg.split_folder, cfg.val_split, cfg.val_split, run)

        # Create train and test dataset objects
        train_dataset = HSIDataset(data.image, train_gt, cfg.sample_size, data_augmentation=True)
        test_dataset = HSIDataset(data.image, test_gt, cfg.sample_size, data_augmentation=False)
        val_dataset = HSIDataset(data.image, val_gt, cfg.sample_size, data_augmentation=False)

        # Create train and test loaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False)

        # Setup model, optimizer and loss
        model = DFFN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)

        # Scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step, gamma=cfg.gamma)

        # Run epochs
        running_loss = 0.0
        running_correct = 0
        total_steps = len(train_loader)
        test_size = len(test_loader)
        for epoch in range(cfg.num_epochs):
            print("RUNNING EPOCH {}/{}".format(epoch + 1, cfg.num_epochs))

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
                lr_scheduler.step()

                # Print steps and loss every PRINT_FREQUENCY
                if (i + 1) % cfg.print_frequency == 0 or i + 1 == total_steps:
                    tqdm.write(
                        f'\tEpoch [{epoch + 1}/{cfg.num_epochs}], Step [{i + 1}/{total_steps}]\tLoss: {loss.item():.4f}')

                # Compute intermediate results for visualization
                if writer is not None:
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    running_correct += (predicted == labels).sum().item()

                    # Write steps and loss every WRITE_FREQUENCY to tensorboard
                    if (i + 1) % cfg.write_frequency == 0:
                        writer.add_scalar('training loss', running_loss / cfg.write_frequency, epoch * total_steps + i)
                        writer.add_scalar('accuracy', running_correct / cfg.write_frequency, epoch * total_steps + i)
                        running_loss = 0.0
                        running_correct = 0

            # Run validation
            if cfg.val_split > 0:
                test_model(model, val_loader, writer)

        print("Finished training!")


# Main function
def main():
    writer = SummaryWriter('tensorboard')
    train()
    writer.close()


if __name__ == '__main__':
    main()
