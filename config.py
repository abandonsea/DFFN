#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 07 19:02 2021

@author: Pedro Vieira @description: Settings for the DFFN training and testing
"""

import shutil
import os
import yaml


class DFFNConfig:
    def __init__(self, filename='config.yaml', test=False):
        # Load config
        with open(filename, "r") as file:
            cfg = yaml.safe_load(file)

            # Dataset settings
            self.dataset = cfg['dataset']
            self.experiment = cfg['experiment']
            self.data_folder = cfg['data_folder']
            self.exec_folder = cfg['exec_folder'] + self.experiment + '/'
            self.split_folder = self.exec_folder + cfg['split_folder'] + self.dataset + '/'
            self.checkpoint_folder = self. exec_folder + cfg['checkpoint_folder'] + self.dataset + '/'
            self.checkpoint_file = cfg['checkpoint_file']
            self.use_checkpoint = cfg['use_checkpoint']
            self.val_split = cfg['val_split']
            self.train_split = cfg['train_split']
            self.train_batch_size = cfg['train_batch_size']
            self.test_batch_size = cfg['test_batch_size']
            self.sample_size = cfg['sample_size']
            self.sample_bands = cfg['sample_bands']
            self.generate_samples = cfg['generate_samples']
            self.max_samples = cfg['max_samples']

            # Hyper parameters
            self.num_runs = cfg['num_runs']
            self.num_epochs = cfg['num_epochs']
            self.learning_rate = cfg['learning_rate']
            self.momentum = cfg['momentum']
            self.weight_decay = float(cfg['weight_decay'])
            self.gamma = cfg['gamma']
            self.scheduler_step = cfg['scheduler_step']

            # Other options
            self.print_frequency = cfg['print_frequency']
            self.write_frequency = cfg['write_frequency']

        # Copy config to execution folder
        if not (test or self.use_checkpoint or self.generate_samples):
            assert not os.path.isdir(self.exec_folder), 'Current experiment name already exists. '\
                                                        'Please provide a new experiment name.'
            os.makedirs(self.exec_folder)
            shutil.copyfile(filename, self.exec_folder + filename)
