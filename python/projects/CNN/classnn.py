"""
Class nn
Accepts numpy arrays as mnist dataset, returns model
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import shutil
import json
from sys import platform

# To predict model without recreating graph for prediction
# Refer to this link


class ClassNN:
    model_dir_pose = '/home/mauricio/models/nn_classifier'
    model_dir_action = '/home/mauricio/models/nn_class_action'
    classes_num_pose = 8
    hidden_num_pose = 40

    if platform == 'win32':
        model_dir_action = 'C:\\models\\nn_class_action'
        model_dir_pose = 'C:\\models\\nn_classifier'

    def __init__(self, model_dir, classes, hidden_number, label_names=list(), learning_rate=0.001):
        self.model_dir = model_dir
        self.classes = classes
        self.hidden_number = hidden_number
        self.label_names = label_names
        self.classifier = tf.estimator.Estimator(
            model_fn=lambda features, labels, mode: self.nn_model_fn(features, labels,
                                                                        mode, classes, hidden_number, learning_rate),
            model_dir=model_dir)
        tf.logging.set_verbosity(tf.logging.INFO)
        self.fast_predict = FastPredict(self.classifier)

    @classmethod
    def load_from_params(cls, model_dir):
        path_params = os.path.join(model_dir, 'params.json')

        if not os.path.isfile(path_params):
            raise Exception('Params file does not exist: {0}'.format(path_params))

        with open(path_params, 'r') as file_text:
            params_str = file_text.read()

        params = json.loads(params_str)
        classes = params['classes']
        hidden_number = params['hidden_number']
        label_names = params['label_names']

        return cls(model_dir, classes, hidden_number, label_names)

    def train_model(self, train_data, train_labels, remove_train_folder=True, label_names=list(),
                    steps=20000):
        print('Init training the model')

        # Remove training folder if exists
        if os.path.exists(self.model_dir) and remove_train_folder:
            print('Removing folder {0}'.format(self.model_dir))
            shutil.rmtree(self.model_dir)

        # Set up logging for predictions
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=50,
            num_epochs=None,
            shuffle=True)

        self.classifier.train(
            input_fn=train_input_fn,
            steps=steps)

        # Saving training parameters
        training_params = {
            'classes': self.classes,
            'hidden_number': self.hidden_number
        }

        if len(label_names) != 0:
            training_params['label_names'] = label_names

        # Write training configuration into file
        training_params_str = json.dumps(training_params)
        training_params_path = os.path.join(self.model_dir, 'params.json')
        with open(training_params_path, 'w') as text_file:
            text_file.write(training_params_str)

        # Done!

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
            raise Exception('Dimension of array is not one. number dim: {0}'.format(predict_data.ndim))

        # Initializing prediction
        array = np.expand_dims(predict_data, axis=0)

        print('Build model')
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': array},  # The dimensionality is preserved ->Checking
            num_epochs=1,
            shuffle=False
        )
        print('Predict model')

        predict_results = self.classifier.predict(input_fn=predict_input_fn)

        # Returns a generator
        print('Reading predictions')
        result = 0
        for prediction in predict_results:
            result = prediction

        # Returned result
        return result

    def predict_model_array(self, predict_data: np.ndarray):
        if predict_data.ndim != 2:
            raise Exception('Dimension of array is not two. number dim: {0}'.format(predict_data.ndim))

        # Initializing prediction
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': predict_data},  # The dimensionality is preserved ->Checking
            num_epochs=1,
            shuffle=False
        )

        predict_results = self.classifier.predict(input_fn=predict_input_fn)

        # Returns a generator
        print('Reading predictions')
        list_predictions = []
        for prediction in predict_results:
            list_predictions.append(prediction)

        # Returned result
        return list_predictions[0]

    def predict_model_fast(self, predict_data: np.ndarray):
        # Return value
        # classes: number
        # probabilities: array
        if predict_data.ndim != 1:
            raise Exception('Dimension of array is not one. number dim: {0}'.format(predict_data.ndim))

        predict_results = self.fast_predict.predict(predict_data)

        # Returns a generator
        list_predictions = []
        for prediction in predict_results:
            list_predictions.append(prediction)

        # Return result
        return list_predictions[0]

    @staticmethod
    def nn_model_fn(features, labels, mode, classes, hidden_number, learning_rate):
        # Defining input layer
        input_layer = features['x']

        # Defining hidden layer
        hidden_neurons = hidden_number
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


class FastPredict:
    def __init__(self, estimator: tf.estimator.Estimator):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.next_features = []
        self.features_size = 0
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
        if feature_batch.ndim != 1:
            raise Exception('NDim is not one: {0}'.format(feature_batch.ndim))

        features = np.expand_dims(feature_batch, axis=0)
        self.next_features = features
        if self.first_run:
            if features.dtype == np.float32:
                self.type = tf.float32
            elif features.dtype == np.float64:
                self.type = tf.float64
            else:
                raise Exception('Type not supported: {0}'.format(features.dtype))

            self.features_size = features.shape[1]
            self.predictions = self.estimator.predict(
                input_fn=self.example_input_fn(self._create_generator))
            self.first_run = False
        elif self.features_size != features.shape[1]:
            raise ValueError("All batches must be of the same size. First-batch:" + str(self.features_size) +
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
            features = tf.reshape(features, [1, self.features_size])

            return {'x': features}

        return _inner_input_fn
