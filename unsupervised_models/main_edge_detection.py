"""
Script to perform edge detection based segmentation.

NOTE: script is more a feasibility study and thus it uses only the labeled data
loaded by the `load_complete_data()` function.
NOTE: parameters for canny and hough_elipse method were set experimentally and
it would probably ned a bit more time for experimenting in order to make it more
robust.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import ellipse
from skimage.feature import canny
from skimage.transform import hough_ellipse
from sklearn.metrics import classification_report, jaccard_similarity_score

from tools.reader import load_complete_data


def parse_args():
    """
    Argument parser

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser('Perform watershed.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the folder with the data')
    parser.add_argument('--perform_visualizations', action='store_true',
                        help='Visualize predictions')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data_path
    visualizations = args.perform_visualizations

    images, o_contours, i_contours = load_complete_data(data_path)

    # mask images for the area within the o-contour, since it's given to us
    for image, mask in zip(images, o_contours):
        image[~mask] = 0

    within_o_contour_mask = images.ravel() > 0
    labels = i_contours.ravel()[within_o_contour_mask]
    relevant_image_pixels = images.ravel()[within_o_contour_mask]
    relevant_image_pixels = images.ravel()[within_o_contour_mask]

    # perform thresholding given the threshold
    predictions = []
    for image, mask in zip(images, o_contours):
        # detect edges using the canny edge detector
        edges = canny(image, sigma=3, low_threshold=0.2, high_threshold=0.8,
                      use_quantiles=True, mask=mask)
        # use hough ellipse in order to fit the expected shape
        result = hough_ellipse(edges, accuracy=4, threshold=1, min_size=10)
        prediction = np.zeros_like(image, dtype='bool')
        if len(result) > 0:
            result.sort(order='accumulator')
            best = list(result[-1])
            yc, xc, a, b = [int(round(x)) for x in best[1:5]]
            orientation = best[5]
            print(yc, xc, a, b, orientation)
        cy, cx = ellipse(yc, xc, a, b, rotation=-orientation)
        prediction[cy, cx] = True
        if visualizations:
            plt.imshow(prediction)
            plt.show()
        predictions.append(prediction)

    # perform evaluation with our i-contour labels
    relevant_predictions = np.array(
        predictions, dtype='bool').ravel()[within_o_contour_mask]
    print(classification_report(
        labels, relevant_predictions,
        target_names=('between contours', 'inside i-contour')))
    print('mean IOU: {}'.format(
        jaccard_similarity_score(labels, relevant_predictions)))
