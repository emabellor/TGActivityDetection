"""
Class nn
Accepts numpy arrays as mnist dataset, returns model
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np


class ClassNN:

    def __init__(self, model_dir, classes):
        self.model_dir = model_dir
        self.classifier = tf.estimator.Estimator(
            model_fn=lambda features, labels, mode: ClassNN.nn_model_fn(features, labels, mode, classes),
            model_dir=model_dir)
        tf.logging.set_verbosity(tf.logging.INFO)

    def train_model(self, train_data, train_labels):
        print('Training model')

        print('Set up logging for predictions')
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

        print('Train the model')
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=50,
            num_epochs=None,
            shuffle=True)

        self.classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])

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
        if predict_data.ndim != 1:
            print('Dimension of array is not one. ndim: ' + str(predict_data.ndim))
            return -1
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

    @staticmethod
    def nn_model_fn(features, labels, mode, classes):
        # Defining input layer
        input_layer = features['x']

        # Defining hidden layer
        hidden_neurons = 100
        hidden_layer = tf.layers.dense(inputs=input_layer, units=hidden_neurons)

        # Defining output layer
        output_layer = tf.layers.dense(inputs=hidden_layer, units=classes)

        predictions = {
            'classes': tf.argmax(input=output_layer, axis=1),
            'probabilities': tf.nn.softmax(output_layer, name='softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_layer)

        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate = 0.001
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Default - Evaluation
        eval_metrics_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=predictions['classes']
            )
        }

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)
