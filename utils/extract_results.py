#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

############
# Set file #
############
PATH = '../../../Results/'
EXPERIMENT = 'full/'
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
    return oa, aa, kappa


# Main for running script independently
def main():
    for data in DATASETS:
        for net in NETWORKS:
            file = 'test_' + data + '_' + net + '.txt'
            filename = PATH + EXPERIMENT + file
            oa, aa, kappa = get_values(filename)

            oa_mean = oa.mean()
            aa_mean = aa.mean()
            kp_mean = kappa.mean()

            print(f'TEST: {net} with {data}')
            print('#' * 15)
            print(f'OA: {oa.mean():.6f} (+- {oa.std():.6f})')
            print(f'AA: {aa.mean():.6f} (+- {aa.std():.6f})')
            print(f'Kappa: {kappa.mean():.6f} (+- {kappa.std():.6f})')
            print('-' * 15)
            print(f'Max OA: {np.max(oa):.5f}')
            print(f'Min OA: {np.min(oa):.5f}')
            print('')


if __name__ == '__main__':
    main()
