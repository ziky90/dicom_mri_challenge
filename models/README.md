# Models

Implementations of few example networks that can be used as a starting point towards the segmentation of DICOM images.

## Available models

### FCN32

FCN32 model based on the [Fully Convolutional Networks for Semantic Segmentation]
(https://arxiv.org/abs/1411.4038)

### simple segmentation CNN

Simple segmentation CNN

`conv->bn->maxpool->conv->bn->maxpool->conv->bn->transposed_conv`