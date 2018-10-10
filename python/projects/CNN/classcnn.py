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
from sys import platform


class ClassCNN:
    model_dir_action = '/home/mauricio/models/cnn_classifier_action'

    if platform == 'win32':
        video_base_path = 'C:\\models\\cnn_classifier_action'

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
        if height % 4 != 0:
            raise Exception('Height must be multiple of 4')

        self.model_dir = model_dir
        self.classifier = tf.estimator.Estimator(
            model_fn=lambda features, labels, mode:
                self.cnn_model_fn(features, labels, mode),
            model_dir=model_dir)
        tf.logging.set_verbosity(tf.logging.INFO)

        self.fast_predict = FastPredict(self.classifier)

    def train_model(self, train_data, train_labels, steps=20000):
        self.train_steps = steps
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

    def predict_model_fast(self, predict_data: np.ndarray):
        # Return value
        # classes: number
        # probabilities: array
        if predict_data.ndim != 3:
            raise Exception('Dimension of array is not three. number dim: {0}'.format(predict_data.ndim))

        predict_results = self.fast_predict.predict(predict_data)

        # Returns a generator
        list_predictions = []
        for prediction in predict_results:
            list_predictions.append(prediction)

        # Return result
        return list_predictions[0]

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


class FastPredict:
    def __init__(self, estimator: tf.estimator.Estimator):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.next_features = []
        self.features_height = 0
        self.features_width = 0
        self.features_dim = 0
        self.predictions = None
        self.type = None

    def _create_generator(self):
        while not self.closed:
            yield self.next_features

    def predict(self, feature_batch: np.ndarray):
        """ Runs a prediction on a set of features. Calling multiple times
            does *not* regenerate the graph which makes predict much faster.
            feature_batch a list of list of features. IMPORTANT: If you're only classifying 1 thing,
            you still need to make it a batch of 1 by wrapping it in a list (i.e. predict([my_feature]), not predict(my_feature)
        """
        # Initializing prediction
        if feature_batch.ndim != 3:
            raise Exception('NDim is not three: {0}'.format(feature_batch.ndim))

        features = np.expand_dims(feature_batch, axis=0)
        self.next_features = features
        if self.first_run:
            if features.dtype == np.float32:
                self.type = tf.float32
            elif features.dtype == np.float64:
                self.type = tf.float64
            else:
                raise Exception('Type not supported: {0}'.format(features.dtype))

            self.features_height = features.shape[1]
            self.features_width = features.shape[2]
            self.features_dim = features.shape[3]

            self.predictions = self.estimator.predict(
                input_fn=self.example_input_fn(self._create_generator))
            self.first_run = False

        elif self.features_height != features.shape[1] or \
                self.features_width != features.shape[2] or \
                self.features_dim != features.shape[3]:
            raise ValueError("All batches must be of the same size. Current-batch:" + str(features.shape) +
                             " This-batch:" + str(features.shape[1]))

        # Only read one size
        results = list()
        results.append(next(self.predictions))
        return results

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. This is probably OK")

    def example_input_fn(self, generator):
        """ An example input function to pass to predict. It must take a generator as input """
        def _inner_input_fn():
            data_set = tf.data.Dataset().from_generator(generator, output_types=self.type).batch(1)
            iterator = data_set.make_one_shot_iterator()
            features = iterator.get_next()

            # Reshape elements to same size that batch_size
            features = tf.reshape(features, [1, self.features_height, self.features_width, self.features_dim])

            return {'x': features}

        return _inner_input_fn

