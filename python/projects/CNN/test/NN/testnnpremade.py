"""
How to use premade estimator with mnist
https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def input_fn(dataset):
    return dataset.images, dataset.labels.astype(np.int32)


def main():
    mnist = input_data.read_data_sets('MNIST_data')
    data, labels = input_fn(mnist.train)

    print('Using estimators by premade')
    print('Loading minst')

    print('Load training and eval data')
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # type:np.ndarray
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # type:np.ndarray
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    my_feature_colums = []
    my_feature_colums.append(tf.feature_column.numeric_column(key='x', shape=[28, 28]))

    print('Loading the premade estimator')
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[100],
        n_classes=10,
        feature_columns=my_feature_colums,
        model_dir='/tmp/model_premade',
        optimizer=tf.train.GradientDescentOptimizer(0.001)
    )

    print('Set up logging for predictions')
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        num_epochs=None,
        batch_size=50,
        shuffle=True
    )

    steps = 20000
    estimator.train(input_fn=train_input_fn,
                    steps=steps)

    print('Setting eval')

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )

    # Evaluate the model.
    eval_result = estimator.evaluate(input_fn=test_input_fn)

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    print('Done!')


if __name__ == '__main__':
    main()
