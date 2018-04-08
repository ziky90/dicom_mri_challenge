"""Script to predict from the CNN model"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tools.parsing import parse_dicom_file


def parse_args():
    """
    Argument parser

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser('Visualize DICOM and contour')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model.')
    parser.add_argument('--dicom_path', type=str, required=True,
                        help='Path to the dicom file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    dicom_path = args.dicom_path

    model = tf.keras.models.load_model(model_path)
    # just a quick script to check prediction for one image
    input_image = np.expand_dims(parse_dicom_file(dicom_path)['pixel_data'], -1)
    prediction = np.squeeze(model.predict(np.expand_dims(input_image, 0)))
    # getting argmax
    prediction_mask = np.argmax(prediction, -1)
    plt.imshow(prediction_mask)
    plt.show()
