"""
Script to define the FCN32 model based on:

Fully Convolutional Networks for Semantic Segmentation
https://arxiv.org/abs/1411.4038
"""

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, Dropout, Conv2DTranspose
from tensorflow.python.keras.applications import VGG16


CLASSES = 2
DROPOUT_RATE = 0.0


def FCN32(input_shape, dropout_rate=0.0, classes=2):
    """
    Implementation of the FCN32 network based on the VGG16 from
    keras.applications.

    :param input_shape: Model input shape.
    :type input_shape: (int, int)
    :param dropout_rate: dropout rate to be used in the model.
    :type dropout_rate: float
    :param classes: Number of classes for the segmantation.
    :type classes: int
    :return: FCN 32 Keras model.
    :rtype: `tensorflow.python.keras.Model`
    """
    net = VGG16(include_top=False, input_shape=input_shape, weights=None)
    inputs = net.input
    base_output = net.output

    net = Conv2D(4096, (7, 7), padding='same', activation='relu')(base_output)
    net = Dropout(dropout_rate)(net)
    net = Conv2D(4096, (1, 1), padding='same', activation='relu')(net)
    net = Dropout(dropout_rate)(net)
    net = Conv2D(classes, (1, 1))(net)
    net = Conv2DTranspose(classes, (64, 64), strides=32, use_bias=False,
                          padding='same', activation='softmax')(net)
    model = Model(inputs, net, name='fcn32')
    # check the model using the summary
    model.summary()
    return model
