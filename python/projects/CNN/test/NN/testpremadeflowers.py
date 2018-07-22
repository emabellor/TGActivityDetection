from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth': np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth': np.array([2.2, 1.0])}

    labels = np.array([2, 1])
    return features, labels


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)


def main():
    print('Using ventosa')

    print('loading datasets')
    train_test = tf.data.Dataset

    my_feature_colums = []
    train_x, train_y = input_evaluation_set()

    # 4 columns, 4 numeric keys
    for key in train_x.keys():
        my_feature_colums.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_colums,
        hidden_units=[10, 10],
        n_classes=3,
        model_dir='/tmp/premade_flowers'
    )


    # Training the model
    batch_size = 10
    train_steps = 1000
    classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, 10), steps=train_steps)

    print('Done!')


if __name__ == '__main__':
    main()

