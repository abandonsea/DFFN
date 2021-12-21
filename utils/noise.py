#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 7 13:31 2021

@author: Pedro Vieira
@description: Implements a function to add noise to a test hyperspectral image
"""

import torch
import numpy as np

from utils.tools import HSIData

# Set image file path
PATH = '../experiments/prototype/'
IMAGE_NAME = 'proc_image.pth'

# Set noise parameters
NOISE_TYPE = 'salt_and_pepper'
NOISE_AMOUNT = 0.1


def add_noise(img, noise_params):
    noise_type = noise_params[0]
    noise_amount = noise_params[1]
    if noise_amount == 0:
        return img

    # Set out data to 0 mean and 1 variance to apply noise, then undo normalization
    out = np.copy(img)
    out, scaler = HSIData.normalize(out, 'standard')

    max_value = np.max(out)
    min_value = np.min(out)

    # Applies a salt and pepper noise to each band for a noise_param ratio of the pixels
    # noise_param = fraction of noisy pixels [0.1)
    if noise_type == 'salt_and_pepper':
        # Calculate the number of image salt and pepper noise points
        pixels_per_band = img.shape[0] * img.shape[1]
        num_points = int(np.floor(pixels_per_band * noise_amount))

        for band in range(img.shape[2]):
            for i in range(num_points):
                rand_x = np.random.randint(img.shape[0])  # Generate random x coordinate
                rand_y = np.random.randint(img.shape[1])  # Generate random y coordinate

                out[rand_x, rand_y, band] = min_value if np.random.random() <= 0.5 else max_value

    # Applies an additive gaussian noise to every pixel with mean and variance defined by noise_param
    # noise_param = sigma; normal = [mu, sigma]
    elif noise_type == 'additive_gaussian':
        out += np.random.normal(0.0, noise_amount, size=img.shape)

    # Applies a multiplicative gaussian noise to every pixel with mean and variance defined by noise_param
    # noise_param = sigma; noise = normal(1.0, noise_param)
    elif noise_type == 'multiplicative_gaussian':
        out = np.multiply(out, np.random.normal(1.0, noise_amount, size=img.shape))

    else:
        raise Exception('Noise type not implemented')

    # Prune out of bounds values
    out_of_bounds_max = np.nonzero(out > max_value)
    out[out_of_bounds_max] = max_value
    out_of_bounds_min = np.nonzero(out < min_value)
    out[out_of_bounds_min] = min_value

    out = HSIData.denormalize(out, scaler)

    return out


# Main for running script independently
def main():
    img = torch.load(PATH + IMAGE_NAME)

    noisy_img = add_noise(img, [NOISE_TYPE, NOISE_AMOUNT])
    torch.save(noisy_img, PATH + 'img_noise_' + NOISE_TYPE + '.pth')


if __name__ == '__main__':
    main()
