#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

############
# Set file #
############
PATH = '../../../Results/'
FILE = 'test_paviau_dffn.txt'
DATASETS = ['paviau', 'indian_pines', 'salinas']
NETWORKS = ['sdmm', 'dffn', 'vscnn', 'sae3ddrn']

VALUE_POSITION = 3


# Get test results from text file
def get_values(filename):
    overall_accuracy = []
    average_accuracy = []
    kappa_coefficients = []

    with open(filename, 'r') as file:
        line = file.readline()
        while line:
            # Check for OA
            if 'OVERALL ACCURACY' in line:
                words = line.split(' ')
                overall_accuracy.append(float(words[VALUE_POSITION]))
            # Check for AA
            elif 'AVERAGE ACCURACY' in line:
                words = line.split(' ')
                average_accuracy.append(float(words[VALUE_POSITION]))
            # Check for kappa
            elif 'KAPPA COEFFICIENT' in line:
                words = line.split(' ')
                kappa_coefficients.append(float(words[VALUE_POSITION]))

            # Get next line
            line = file.readline()

    assert len(overall_accuracy) == len(average_accuracy), 'Wrong list lengths! [1]'
    assert len(average_accuracy) == len(kappa_coefficients), 'Wrong list lengths! [2]'

    oa = np.array(overall_accuracy)
    aa = np.array(average_accuracy)
    kappa = np.array(kappa_coefficients)
    return oa.mean(), aa.mean(), kappa.mean(), np.max(oa), np.min(oa)


# Main for running test independently
def main():
    for data in DATASETS:
        for net in NETWORKS:
            file = 'test_' + data + '_' + net + '.txt'
            filename = PATH + file
            oa, aa, kappa, a_max, a_min = get_values(filename)

            print(f'TEST: {net} with {data}')
            print('#' * 15)
            print(f'OA: {oa:.6f}')
            print(f'AA: {aa:.6f}')
            print(f'Kappa: {kappa:.6f}')
            print('-' * 15)
            print(f'Max OA: {a_max:.5f}')
            print(f'Min OA: {a_min:.5f}')
            print('')


if __name__ == '__main__':
    main()
