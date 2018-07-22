from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""

    # Input Layer
    # We want to reshape our input images to 28x28
    # The first argument is batch_size. When this argument is -1, the value of batch size is computed dynamically
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    print('Labels: ', labels)

    # Convolutional Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    # Flat array in two dimensional - batch_size, width * height * layetrs
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7, 64])

    # Create dense layer - 1024 neurons
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Create regularization - dropout - Activate neuron with some probability
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer - with a confidence value!!
    logits = tf.layers.dense(inputs=dropout, units= 10)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilites': tf.nn.softmax(logits, name='softmax_tensor')
    }

    # Predict phase
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Train phase
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes']
        )
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    """Main function"""
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')

    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir='/tmp/mnist_convnet_model')

    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    print('Done!')


tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    tf.app.run()