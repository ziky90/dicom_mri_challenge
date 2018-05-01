"""
Script to perform watershed based segmentation.

NOTE: script is more a feasibility study and thus it uses only the labeled data
loaded by the `load_complete_data()` function.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import watershed
from sklearn.metrics import classification_report, jaccard_similarity_score

from tools.reader import load_complete_data


def parse_args():
    """
    Argument parser

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser('Perform watershed.')
    parser.add_argument('--threshold', type=int, required=False, default=None,
                        help='Threshold to be used')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the folder with the data')
    parser.add_argument('--perform_visualizations', action='store_true',
                        help='Visualize predictions')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data_path
    visualizations = args.perform_visualizations
    threshold = args.threshold

    images, o_contours, i_contours = load_complete_data(data_path)

    # mask images for the area within the o-contour, since it's given to us
    for image, mask in zip(images, o_contours):
        image[~mask] = 0

    within_o_contour_mask = images.ravel() > 0
    labels = i_contours.ravel()[within_o_contour_mask]
    relevant_image_pixels = images.ravel()[within_o_contour_mask]
    relevant_image_pixels = images.ravel()[within_o_contour_mask]
    if threshold is None:
        threshold = threshold_otsu(relevant_image_pixels)
        print('automatically computed threshold = {}'.format(threshold))

    # perform thresholding given the threshold
    predictions = []
    for image, mask in zip(images, o_contours):
        markers = np.zeros(image.shape, dtype=np.uint)
        markers[image < threshold] = 0
        markers[image > threshold] = 1
        prediction = watershed(image, markers, mask=mask)
        if visualizations:
            plt.imshow(prediction)
            plt.show()
        predictions.append(prediction)

    # perform evaluation with our i-contour labels
    # NOTE: since the threshold was derived from exactly the same dataset,
    # we're kind of cheating and we should treat this as an evaluation on the
    # training data!
    relevant_predictions = np.array(
        predictions, dtype='bool').ravel()[within_o_contour_mask]
    print(classification_report(
        labels, relevant_predictions,
        target_names=('between contours', 'inside i-contour')))
    print('mean IOU: {}'.format(
        jaccard_similarity_score(labels, relevant_predictions)))
