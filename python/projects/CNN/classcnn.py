"""
Class CNN
Convolutional neural networks
Example taken from tensorflow
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import shutil
import os


class ClassCNN:
    model_dir_action = '/home/mauricio/models/cnn_classifier_action'

    def __init__(self,
                 model_dir,
                 classes,
                 width,
                 height,
                 channels,
                 train_steps=20000,
                 batch_size=64):

        self.classes = classes
        self.width = width
        self.height = height
        self.channels = channels
        self.train_steps = train_steps
        self.batch_size = batch_size

        if width % 4 != 0:
            raise Exception('Width must be multiple of 4')
        elif height % 4 != 0:
            raise Exception('Height must be multiple of 4')
        else:
            self.model_dir = model_dir
            self.classifier = tf.estimator.Estimator(
                model_fn=lambda features, labels, mode:
                    self.cnn_model_fn(features, labels, mode),
                model_dir=model_dir)
            tf.logging.set_verbosity(tf.logging.INFO)

    def train_model(self, train_data, train_labels):
        print('Training model')

        print('Remove training folder if exists -> Avoid confusions')
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)

        print('Set up logging for predictions')
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

        print('Train the model')
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)

        self.classifier.train(
            input_fn=train_input_fn,
            steps=self.train_steps,
            hooks=[logging_hook])

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def eval_model(self, eval_data, eval_labels):
        print('Evaluate the model and print results')
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_results))

    def predict_model(self, predict_data: np.ndarray):
        # Check dimensionality first
        if predict_data.ndim != 3:
            raise Exception('Dimension of array is not 3. ndim: {0}'.format(predict_data.ndim))
        else:
            print('Initializing prediction')
            array = np.expand_dims(predict_data, axis=0)
            print(array.shape)

            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': array},  # The dimensionality is preserved ->Checking
                num_epochs=1,
                shuffle=False
            )

            predict_results = self.classifier.predict(input_fn=predict_input_fn)

            # Returns a generator
            print('Reading predictions')
            result = 0
            for prediction in predict_results:
                result = prediction

            # Returned result
            return result

    """
    ConvNet 32 filters first layer, 64 filters second layer
    1024 Neurons dense layer
    Support multichannel from images
    """
    def cnn_model_fn(self, features, labels, mode):
        # Input layer
        input_layer = tf.convert_to_tensor(features['x'])

        # Convolutional layer 1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )

        # Pooling layer 1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional layer 2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )

        # Pooling layer 2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense layer
        # TODO -> Check dimension with the number of channels!
        pool2_flat_width = int(self.width / 4 * self.height / 4 * 64)
        pool2_flat = tf.reshape(pool2, [-1, pool2_flat_width])
        dense = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
            activation=tf.nn.relu
        )

        # Dropout
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits layer
        logits = tf.layers.dense(inputs=dropout, units=self.classes)

        # Predictions, use argmax function
        predictions = {
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the training optimizer
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Eval mode by default
        # Add evaluation metrics
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

