#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 5 17:57 2021

@author: Pedro Vieira
@description: Implements the test function for the DFFN network published in https://github.com/weiweisong415/Demo_DFFN_for_TGRS2018
"""

import torch
import torch.utils.data as Torchdata
import numpy as np
from tqdm import tqdm

from tools import *
from net import *

# Parameters setting
DATASET = 'PaviaU'  # PaviaU; Salinas; KSC
FOLDER = './Datasets/'  # the dataset folder
CHECKPOINT_FOLDER = 'checkpoints/' + DATASET + '/'


# Test DFFN
def test():
    print('Dummy!')


# Main for running test independently
def main():
    test()


if __name__ == '__main__':
    main()
