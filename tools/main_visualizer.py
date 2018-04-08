"""Visualizer in order to visualize images and labels"""

import argparse

import matplotlib.pyplot as plt

from tools.parsing import parse_dicom_file, poly_to_mask, parse_contour_file


def parse_args():
    """
    Argument parser

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser('Visualize DICOM and contour')
    parser.add_argument('--dicom_path', type=str, required=False, default=None,
                        help='Path to the DICOM file.')
    parser.add_argument('--contour_path', type=str, required=False, default=None,
                        help='Path to the contour file.')
    return parser.parse_args()


def visualize_dicom(dicom_path):
    """
    Visualize the DICOM image.

    :param dicom_path: Path to the DICOM file.
    :type dicom_path: str
    """
    image = parse_dicom_file(dicom_path)['pixel_data']
    plt.imshow(image)
    plt.show()


def visualize_contour_mask(contour_path, width=256, height=256):
    """
    Visualize the contour converted to mask.

    :param contour_path: Path to the contour file.
    :type contour_path: str
    :param width: Width of the contour.
    :type width: int
    :param height: Height of the contour.
    :type height: int
    """
    image = poly_to_mask(parse_contour_file(contour_path), width, height)
    plt.imshow(image)
    plt.show()


def visualize_dicom_with_contour(dicom_path, contour_path):
    """
    Visualize the DICOM and contour mask.

    :param dicom_path: Path to the DICOM image.
    :type dicom_path: str
    :param contour_path: Path to the corresponding contour file.
    :type contour_path: str
    """
    dicom_image = parse_dicom_file(dicom_path)['pixel_data']
    height, width = dicom_image.shape
    contour_image = poly_to_mask(parse_contour_file(contour_path), width, height)
    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(dicom_image)
    fig.add_subplot(1, 2, 2)
    plt.imshow(contour_image)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    dicom_path = args.dicom_path
    contour_path = args.contour_path

    if dicom_path is not None and contour_path is not None:
        visualize_dicom_with_contour(dicom_path, contour_path)
    if dicom_path is not None:
        visualize_dicom(dicom_path)
    if contour_path is not None:
        visualize_contour_mask(contour_path)
