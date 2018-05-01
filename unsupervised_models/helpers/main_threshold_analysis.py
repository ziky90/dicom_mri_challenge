"""Script to perform analysis for thresholding"""

import argparse
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from tools.reader import load_complete_data


def parse_args():
    """
    Argument parser

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser('Perform thresholding analysis')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the folder with the data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data_path

    images, o_contours, i_contours = load_complete_data(data_path)

    # mask images for the area within the i-contour
    masked_inner_images = []
    for image, mask in zip(deepcopy(images), i_contours):
        image[~mask] = 0
        masked_inner_images.append(image)

    # mask the image for the area between the i-contour and o-contour
    masked_border = []
    for image, i_mask, o_mask in zip(deepcopy(images), i_contours, o_contours):
        image[~(i_mask ^ o_mask)] = 0
        masked_border.append(image)

    inner_values = np.array(masked_inner_images).ravel()
    inner_values_hist_mask = inner_values != 0
    inner_hist_values = inner_values[inner_values_hist_mask]

    border_values = np.array(masked_border).ravel()
    border_values_hist_mask = border_values != 0
    border_hist_values = border_values[border_values_hist_mask]

    bins = max(np.max(inner_hist_values), np.max(border_hist_values))

    plt.hist(inner_hist_values, bins, alpha=0.5, label='inside i-contour')
    plt.hist(border_hist_values, bins, alpha=0.5, label='between contours')
    plt.legend(loc='upper right')
    plt.show()
