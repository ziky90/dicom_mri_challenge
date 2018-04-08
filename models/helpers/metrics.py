"""Script to define the FCN32 model based on"""

import tensorflow as tf

# NOTE right now we have always just 2 classes.
NUM_CLASSES = 2


def mean_iou(y_true, y_pred):
    """
    Custom mean IOU metric for the model training.

    :return: computed mean IOU
    :rtype: float
    """
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, NUM_CLASSES)
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score
