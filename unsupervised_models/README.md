# Unsupervised models

## Thresholding
Thresholding is the simplest method that can be possibly used for the task of semantic segmentation, especially when we have only one channel images that we'd like to separate to two classes. I our case we're trying to separate images to either the `blood pool (inside the i-contour)` or `heart muscle (between the i-contours and o-contours)`.

### Finding threshold
In order to perform thresholding we need to have a look into the data, especially into the intensities in both classes. We can visualize both classes intensities by:

`python helpers/main_threshold_analysis.py --data_path <data_path>`

![intensities histogram](https://github.com/ziky90/dicom_mri_challenge/blob/master/resources/thresholding_histogram.png)
intensities per class histogram

### Thresholding itself
We can run our thresholding script by:

`python main_thresholding.py --data_path <data_path> --threshold 120`

NOTE: if threshold is not provided, Otsu's algorithm is used in order to compute the threshold automatically.

As the result of the evaluation we'll obtain following evaluation:
```
                  precision    recall  f1-score   support

between contours       0.78      0.68      0.73     37714
inside i-contour       0.81      0.87      0.84     58404

     avg / total       0.80      0.80      0.80     96118

mean IOU: 0.798893027321
```
NOTE: since the threshold was derived from exactly the same dataset, we're kind of cheating and we should treat this as an evaluation on the training data!

![thresholded prediction 1](https://github.com/ziky90/dicom_mri_challenge/blob/master/resources/thresholding_prediction1.png)
Cherry picked prediction 1 using threshold 120

![thresholded prediction 2](https://github.com/ziky90/dicom_mri_challenge/blob/master/resources/thresholding_prediction2.png)
Cherry picked prediction 2 using threshold 120
