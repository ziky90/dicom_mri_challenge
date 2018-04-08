"""Test parsing tools"""

import os
import unittest

import numpy as np

from tools.parsing import parse_contour_file, poly_to_mask, parse_dicom_file

LOCAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))


class TestParse(unittest.TestCase):

    def test_dicom_file_to_npz(self):
        """
        Test load DICOM file as npz.
        """
        file_path = os.path.join(LOCAL_PATH, 'test_input', '100.dcm')
        result = parse_dicom_file(file_path)['pixel_data']
        expected = np.load(os.path.join(LOCAL_PATH, 'test_output',
                                        '100_dcm.npy'))
        np.testing.assert_equal(result, expected)

    def test_contour_file_to_mask(self):
        """
        Test parse contour file.
        """
        file_path = os.path.join(LOCAL_PATH, 'test_input',
                                 'IM-0001-0048-icontour-manual.txt')
        result = poly_to_mask(parse_contour_file(file_path), 256, 256)
        expected = np.load(os.path.join(LOCAL_PATH, 'test_output',
                                        'IM-0001-0048-icontour-manual.npy'))
        np.testing.assert_equal(result, expected)
