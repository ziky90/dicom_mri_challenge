# DICOM MRI data challenge



## Requirements

### GIT LFS
In order to be able to download the data for testing and to run the examples, you'll need to install the GIT LFS.
Installation details can be found [here](https://git-lfs.github.com/)

### Python packages
All the needed packages are listed in the `requirements.txt` file. NOTE the repository works only with `pydicom` < 1.

## DICOM data basic description

### Image data
The dataset consists of the DICOM MRI images, usually one folder of .dcm files represents one record and one .dcm file represents one slice of the 3D scan.

### Labels data
Labels corresponding to the images considts of 2 different types i-contour (inner contours) and o-contours (couter contours). **For purposes of this challenge we use only i-contour labels!**

## Parsing and loading DICOM images and it's corresponding Contour files

### Visualization of the data
In order to verify if we're parsing and loading the data correctly, we can use the `tools/main_visualizer.py` script.

Example visualizations can be obtained by:

`python tools/main_visualizer.py --dicom_path tools/visualization_examples/48.dcm --contour_path tools/visualization_examples/IM-0001-0048-icontour-manual.txt`

or by

`python tools/main_visualizer.py --dicom_path tools/visualization_examples/188.dcm --contour_path tools/visualization_examples/IM-0001-0188-icontour-manual.txt`

or in order to obtain the diff of o-contour and i-contour by:

`python tools/main_visualizer.py --contour_path tools/visualization_examples/IM-0001-0099-icontour-manual.txt --o_contour_path tools/visualization_examples/IM-0001-0099-ocontour-manual.txt` 

![visualized dicom image](https://github.com/ziky90/dicom_mri_challenge/blob/master/resources/dicom_188.png)
visualized example DICOM image

![visualized contour image](https://github.com/ziky90/dicom_mri_challenge/blob/master/resources/contour_188.png)
visualized example contour mask image

![visualized dicom_and:contour image](https://github.com/ziky90/dicom_mri_challenge/blob/master/resources/dicom_contour_188.png)
visualized example DICOM and contour mask image in one figure

![visualized contours_diff](https://github.com/ziky90/dicom_mri_challenge/blob/master/resources/contours_99_diff.png)
visualized example o-contour and i-contour diff

### Unit tests for the input files

There were added unit tests for parsing of both the DICOM image input file and the contour label file.

## Data reading pipeline
NOTE: The dataset reading pipeline currently works only with data that fits to the RAM memory. This is faster right now since we have only a small labeled dataset, in case that bigger labeled dataset would be available reader can be easily rewritten as a generator. Second possible option since `tensorflow==1.8.0` tf.data.Dataset should work with Keras models, so since tensorflow==1.8.0 tf.data.Dataset is recommended way to input the data.

## Model training pipeline
NOTE: Model training currently works only either on 1CPU or 1GPU. Simple extension leveraging `tf.keras.utils.multi_gpu_model` can be implemented in urder to train the model on multiple GPUs.

Model training pipeline can be run by script `main_train_model.py`.

Example command:
`python main_train_model.py --network simple_cnn --train_data_path <path_to_data> --model_path model --epochs 10`

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 256, 256, 1)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 256, 256, 128)     1280
_________________________________________________________________
batch_normalization_1 (Batch (None, 256, 256, 128)     512
_________________________________________________________________
activation_1 (Activation)    (None, 256, 256, 128)     0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 128, 128, 128)     0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 128, 128, 256)     295168
_________________________________________________________________
batch_normalization_2 (Batch (None, 128, 128, 256)     1024
_________________________________________________________________
activation_2 (Activation)    (None, 128, 128, 256)     0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 64, 64, 256)       0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 64, 64, 512)       1180160
_________________________________________________________________
batch_normalization_3 (Batch (None, 64, 64, 512)       2048
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 256, 256, 2)       9216
=================================================================
Total params: 1,489,408
Trainable params: 1,487,616
Non-trainable params: 1,792
_________________________________________________________________
Epoch 1/10
2018-04-08 23:46:25.364234: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
4/4 [==============================] - 137s 34s/step - loss: 0.7976 - acc: 0.7068 - val_loss: 1.0160 - val_acc: 0.7075
Epoch 2/10
4/4 [==============================] - 142s 35s/step - loss: 0.6995 - acc: 0.8311 - val_loss: 0.7684 - val_acc: 0.7174
Epoch 3/10
4/4 [==============================] - 146s 37s/step - loss: 0.5913 - acc: 0.9382 - val_loss: 0.4719 - val_acc: 0.9817
Epoch 4/10
4/4 [==============================] - 160s 40s/step - loss: 0.4697 - acc: 0.9797 - val_loss: 0.3573 - val_acc: 0.9820
Epoch 5/10
4/4 [==============================] - 152s 38s/step - loss: 0.3793 - acc: 0.9816 - val_loss: 0.3459 - val_acc: 0.9821
Epoch 6/10
4/4 [==============================] - 154s 39s/step - loss: 0.3445 - acc: 0.9822 - val_loss: 0.3515 - val_acc: 0.9817
Epoch 7/10
4/4 [==============================] - 130s 33s/step - loss: 0.3339 - acc: 0.9825 - val_loss: 0.3578 - val_acc: 0.9819
Epoch 8/10
4/4 [==============================] - 133s 33s/step - loss: 0.3304 - acc: 0.9827 - val_loss: 0.3612 - val_acc: 0.9803
Epoch 9/10
4/4 [==============================] - 137s 34s/step - loss: 0.3299 - acc: 0.9826 - val_loss: 0.3628 - val_acc: 0.9812
Epoch 10/10
4/4 [==============================] - 139s 35s/step - loss: 0.3274 - acc: 0.9833 - val_loss: 0.3646 - val_acc: 0.9785
```
Experiment with simple_cnn (training on CPU):

### Available models

Currently available models are more in detail described [here](https://github.com/ziky90/dicom_mri_challenge/blob/master/models/README.md).

## Prediction
Prediction from the keras model can be done simply by the script `python main_predict_model.py`.

## Unsupervised baseline models

There were performed several unsupervised experiment in order to create sufficient baseline model. More details about unsupervised models can be found in [here](https://github.com/ziky90/dicom_mri_challenge/blob/master/unsupervised_models/README.md).
