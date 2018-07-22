"""
Positive and negative dir must have Eval and Train folders to work without problems
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
import os

width_mnist = 28
height_mnist = 28

class ClassImageDataSet:
    def __init__(self, positive_dir, negative_dir, width_resize, height_resize):
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.width_resize = width_resize
        self.height_resize = height_resize

    def load_train_set(self, load_color=False):
        print('Load training set')
        pos_dir_train = os.path.join(self.positive_dir, 'Train')
        neg_dir_train = os.path.join(self.negative_dir, 'Train')
        return self._load_data_set(pos_dir_train, neg_dir_train, load_color)

    def load_eval_set(self, load_color=False):
        print('Load evaluation set')
        pos_dir_eval = os.path.join(self.positive_dir, 'Eval')
        neg_dir_eval = os.path.join(self.negative_dir, 'Eval')
        return self._load_data_set(pos_dir_eval, neg_dir_eval, load_color)

    @staticmethod
    def load_train_mnist(reshape=False):
        print('Loading train mnist')
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")

        # Reshape mnist to fit the data model in the ConvNet
        train_data = mnist.train.images  # type:np.ndarray

        if reshape:
            train_data = train_data.reshape([-1, width_mnist, height_mnist, 1])

        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

        return train_data, train_labels

    @staticmethod
    def load_eval_mnist(reshape=False):
        print('Loading eval mnist')
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")

        # Reshape mnist to fit the data model in the ConvNet
        eval_data = mnist.test.images  # type:np.ndarray

        if reshape:
            eval_data = eval_data.reshape([-1, width_mnist, height_mnist, 1])

        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        return eval_data, eval_labels

    """
    Load dataset
    Returns numpy arrays of size total_examples, width_resize * height_resize
    """
    def _load_data_set(self, pos_folder, neg_folder, load_color):
        print('Loading data set')

        data_array = []
        labels_array = []

        for file in os.listdir(pos_folder):
            full_path = os.path.join(pos_folder, file)

            if not load_color:
                image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)  # type: np.ndarray
            else:
                image = cv2.imread(full_path, cv2.IMREAD_COLOR)

            if image is None:
                raise Exception('Error reading image {0}'.format(full_path))
            else:
                data_array.append(self._prepare_np_array(image))
                labels_array.append(1)

        for file in os.listdir(neg_folder):
            full_path = os.path.join(neg_folder, file)

            if not load_color:
                image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(full_path, cv2.IMREAD_COLOR)

            if image is None:
                raise Exception('Error reading image '.format(full_path))
            else:
                data_array.append(self._prepare_np_array(image))
                labels_array.append(0)

        data_array = np.array(data_array)
        labels_array = np.asanyarray(labels_array, dtype=np.int32)

        print('Printing returning objects type')
        print(data_array.shape)
        print(labels_array.shape)

        return data_array, labels_array

    def _prepare_np_array(self, image):
        # Don't do squeeze -> Only resizing image
        image_res = cv2.resize(image, (self.width_resize, self.height_resize))  # type: np.ndarray

        if len(image_res.shape) == 2:
            # Add new axis of image
            image_res = image_res[..., np.newaxis]

        return image_res.astype(float) / 255
