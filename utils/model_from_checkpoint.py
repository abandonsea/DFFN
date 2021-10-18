#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:36 2021

@author: Pedro Vieira @description: Implements a function to get the model from a checkpoint dictionary
"""

import torch


#########################
# SET CHECKPOINT VALUES #
#########################
RUN = 10
EPOCH = 10
FOLDER = './experiments/server_01/checkpoints/'  # Replace folder with the correct experiment
FILENAME = 'model_from_checkpoint_run_' + str(RUN) + '_epoch_' + str(EPOCH) + '.pth'
# FILENAME = None  # Write your own filename


def model_from_checkpoint(run, epoch, folder, filename=None):
    # Load checkpoint
    checkpoint_file = folder + 'checkpoint_run_' + str(run) + '_epoch_' + str(epoch) + '.pth'
    loaded_checkpoint = torch.load(checkpoint_file)

    model_state = loaded_checkpoint['model_state']

    if filename is not None:
        torch.save(model_state, filename)

    return model_state


# Main for running test independently
def main():
    model_from_checkpoint(RUN, EPOCH, FOLDER, FILENAME)


if __name__ == '__main__':
    main()
