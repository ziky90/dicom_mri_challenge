"""Script to train the CNN model"""

import argparse
import os

import tensorflow as tf

from models.fcn32 import FCN32
from models.helpers.metrics import mean_iou
from models.simple_segmentation_cnn import simple_cnn
from tools.reader import load_data_as_dataset

# we currently have only binary classification
CLASSES = 2


def parse_args():
    """
    Argument parser

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser('Train the CNN model')
    parser.add_argument('--batch_size', type=int, required=False, default=24,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, required=False, default=1,
                        help='Desired number of epochs.')
    parser.add_argument('--learning_rate', type=float, required=False,
                        default=0.005, help='Learning rate.')
    parser.add_argument('--network', type=str, required=False, default='fcn32',
                        choices=['simple_cnn', 'fcn32'],
                        help='Type of the network to be used')
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to the folder with train data')
    parser.add_argument('--validation_data_path', type=str, required=False,
                        default=None,
                        help='Path to the folder with validation data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path where model and weights will be stored')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    network = args.network
    train_data_path = args.train_data_path
    validation_data_path = args.validation_data_path
    model_path = args.model_path

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True)

    # prepare the train dataset
    train_dataset_x, train_dataset_y = load_data_as_dataset(train_data_path)
    train_dataset_y = tf.keras.utils.to_categorical(train_dataset_y, CLASSES)
    datagen.fit(train_dataset_x)

    input_shape = train_dataset_x.shape[1:]

    validation_data = None
    # if validation data path is provided, load the validation data
    if validation_data_path is not None:
        validation_dataset_x, validation_dataset_y = load_data_as_dataset(
            validation_data_path)
        validation_dataset_y = tf.keras.utils.to_categorical(
            validation_dataset_y, CLASSES)

    if network == 'fcn32':
        model = FCN32(input_shape)
    elif network == 'simple_cnn':
        model = simple_cnn(input_shape)
    else:
        raise NotImplementedError('Network {} is currently not implemented'
                                  ''.format(network))

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['acc', mean_iou])

    model.fit_generator(
        datagen.flow(train_dataset_x, train_dataset_y, batch_size=batch_size),
        shuffle=True, epochs=epochs,
        steps_per_epoch=len(train_dataset_x)/batch_size,
        validation_data=datagen.flow(validation_dataset_x, validation_dataset_y,
                                     batch_size=batch_size))

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # save the model
    model.save(os.path.join(model_path, 'model.h5'))
