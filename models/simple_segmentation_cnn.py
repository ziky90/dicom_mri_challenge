"""Script to define simple baseline segmenattion CNN"""

from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import (
    Conv2D, Activation, BatchNormalization, MaxPool2D, Conv2DTranspose)


def simple_cnn(input_shape, classes=2):

    input = Input(shape=input_shape)
    net = Conv2D(128, (3, 3), padding='same', activation='relu')(input)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = MaxPool2D((2, 2), padding='same')(net)
    net = Conv2D(256, (3, 3), padding='same', activation='relu')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = MaxPool2D((2, 2), padding='same')(net)
    net = Conv2D(512, (3, 3), padding='same', activation='relu')(net)
    net = BatchNormalization()(net)
    net = Conv2DTranspose(classes, (3, 3), strides=4, use_bias=False,
                          padding='same', activation='softmax')(net)
    model = Model(input, net, name='fcn32')
    # check the model using the summary
    model.summary()
    return model
