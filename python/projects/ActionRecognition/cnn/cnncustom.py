"""
Written by
Eder Mauricio Abello Rodriguez
"""

# Imports
import numpy as np
import tensorflow as tf
import iohandler
import tfhandler as tfh
import utils

tf.logging.set_verbosity(tf.logging.INFO)


def train_model(features, labels, mode):
    """Train rgb model"""
    print('Training model')

    # Import numpy data
    input_layer = tf.reshape(features['x'], [-1, 64, 64, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    """Main function"""
    print('Initializing main function - Training')

    print('Loading data')
    file_dir = '/home/mauricio/folderdu'
    files = iohandler.FileHandler.get_all_files(file_dir)

    print('Total image size: ' + str(files.__len__()))
    numpy_list = tfh.TFHandler.get_numpy_array(files)

    print('Numpy list length: ', str(len(numpy_list)))
    labels = []

    for file in files:
        elements = file.split('%')
        id_class = int(elements[len(elements) - 2])
        labels.append(id_class) # Assuming same order

    print('List ', files)
    print('Labels ', labels)
    np_labels = np.asarray(labels, dtype=np.int32)

    print('Create the classifier')
    activity_classifier = tf.estimator.Estimator(model_fn=train_model, model_dir="/tmp/activity_convnet_model")

    print('Set up logging for predictions')
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    print('Train the model')
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": numpy_list},
        y = np_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)


    activity_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook])

    print('Done train')


if __name__ == '__main__':
    main()
