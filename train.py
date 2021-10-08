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
from hsi_dataset import HSIDataset
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

    # Load a checkpoint
    if cfg.use_checkpoint:
        model_state, optimizer_state, scheduler_state, value_states = load_checkpoint(cfg.checkpoint_folder,
                                                                                      cfg.checkpoint_file)
        first_run, first_epoch, loss_state, correct_state = value_states
    else:
        first_run, first_epoch, loss_state, correct_state = (0, 0, 0.0, 0)
        model_state, optimizer_state, scheduler_state = None, None, None

        # Save data for tests if we are not loading a checkpoint
        data.save_data(cfg.exec_folder)

    # Run training
    for run in range(first_run, cfg.num_runs):
        print("Running an experiment with run {}/{}".format(run + 1, cfg.num_runs))

        # Generate samples or read existing samples
        # TODO: Deal with loading data when loading checkpoints
        if cfg.generate_samples:
            train_gt, test_gt, val_gt = data.sample_dataset(cfg.train_split, cfg.val_split, cfg.max_samples)
            HSIData.save_samples(train_gt, test_gt, val_gt, cfg.split_folder, cfg.train_split, cfg.val_split, run)
        else:
            train_gt, _, val_gt = HSIData.load_samples(cfg.split_folder, cfg.train_split, cfg.val_split, run)

        # Create train and test dataset objects
        train_dataset = HSIDataset(data.image, train_gt, cfg.sample_size, data_augmentation=True)
        val_dataset = HSIDataset(data.image, val_gt, cfg.sample_size, data_augmentation=False)

        # Create train and test loaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False)

        # Setup model, optimizer, loss and scheduler
        model = DFFN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step, gamma=cfg.gamma)

        # Start counting loss and correct predictions
        running_loss = 0.0
        running_correct = 0

        # Load variable states when loading a checkpoint
        if cfg.use_checkpoint:
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            lr_scheduler.load_state_dict(scheduler_state)
            running_loss = loss_state
            running_correct = correct_state

        # Run epochs
        model = model.to(device)
        total_steps = len(train_loader)
        for epoch in range(first_epoch, cfg.num_epochs):
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

            # Save checkpoint
            first_epoch = 0
            checkpoint = {
                'run': run,
                'epoch': epoch,
                'loss_state': running_loss,
                'correct_state': running_correct,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, 'checkpoint_run_' + str(run) + '_epoch_' + str(epoch) + '.pth')

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
