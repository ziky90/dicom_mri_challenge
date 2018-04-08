"""Loading the data for the model training"""

import csv
import glob
import os

import numpy as np

from tools.parsing import parse_contour_file, parse_dicom_file, poly_to_mask

LINK_CSV_FILE = 'link.csv'
DICOMS_FOLDER_NAME = 'dicoms'
COUNTOURS_FOLDER_NAME = 'contourfiles'
I_CONTOURS_FOLDER_NAME = 'i-contours'
CONTOURS_EXTENSION_PATTERN = '*.txt'
DICOM_EXTENSION = '.dcm'

BUFFER_SIZE = 100


def parse_csv_mapping_file(data_path, csv_file_path=LINK_CSV_FILE):
    """
    Parse DICOM -> countour ids mapping .csv file.

    :param data_path: Path to the folder with the DICOM data and labels.
    :type data_path: str
    :param csv_file_path:
    :type csv_file_path:
    :return:
    :rtype:
    """
    ids_mapping = {}
    with open(os.path.join(data_path, csv_file_path), 'rb') as csvfile:
        links = csv.reader(csvfile, delimiter=',')
        for pos, row in enumerate(links):
            # skip the header row
            if pos == 0:
                continue
            ids_mapping[row[0]] = row[1]
    return ids_mapping


def contour_name_to_dicom_name(contour_name):
    """
    Convert contour file name to the DICOM file name.

    :param contour_name: Contour file name.
    :type contour_name: str
    :return: Corresponding DICOM file name.
    :rtype: str
    """
    dicom_number = int(contour_name.split('-')[2])
    return str(dicom_number) + DICOM_EXTENSION


def load_data_as_dataset(data_path):
    """
    Load the data.

    NOTE: assumes that all the data can fit to the RAM memory.
    NOTE: for the purpose of the challenge only the i-contour data is used
    NOTE: only DICOM files having the corresponding i-contour file are loaded

    :param data_path: Path to the folder with the DICOM data and labels.
    :type data_path: str
    :return: images, labels
    :rtype: np.ndarray.Dataset, np.ndarray.Dataset
    """
    images = []
    labels = []
    ids_mapping = parse_csv_mapping_file(data_path)
    for dicom, contour in ids_mapping.iteritems():
        contour_files_path = os.path.join(data_path, COUNTOURS_FOLDER_NAME,
                                          contour, I_CONTOURS_FOLDER_NAME)
        dicom_files_path = os.path.join(data_path, DICOMS_FOLDER_NAME, dicom)
        for contour_file_path in glob.glob(os.path.join(
                contour_files_path, CONTOURS_EXTENSION_PATTERN)):
            contour_list = parse_contour_file(contour_file_path)
            dicom_file_name = contour_name_to_dicom_name(
                os.path.basename(contour_file_path))
            dicom_pixels = parse_dicom_file(os.path.join(
                dicom_files_path, dicom_file_name))['pixel_data']
            height, width = dicom_pixels.shape
            images.append(dicom_pixels)
            labels.append(poly_to_mask(contour_list, width, height))
    images = np.array(images)
    images = np.expand_dims(images, -1)
    labels = np.array(labels)
    assert images.shape[0] == labels.shape[0], \
        'There must be exactly the same number of DICOMS and contours.'
    return images, labels
