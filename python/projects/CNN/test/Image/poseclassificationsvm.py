from classutils import ClassUtils
import os
import cv2
from classopenpose import ClassOpenPose
from classdescriptors import ClassDescriptors
from classloaddescriptors import ClassLoadDescriptors, EnumDesc
import json
from classnn import ClassNN
import numpy as np
import random
from tkinter import Tk
from enum import Enum
import math


def main():
    print('Initializing main function')
    res = input('Press 1 to get all points')

    if res == '1':
        type_desc = EnumDesc.ALL_TRANSFORMED
        load_descriptors(type_desc)
    else:
        raise Exception('Option not implemented')


def load_descriptors(type_desc: EnumDesc):
    print('Loading descriptors using type: {0}'.format(type_desc))

    # Generating elements into list
    results = ClassLoadDescriptors.load_pose_descriptors(type_desc)

    training_data = results['trainingData']
    training_labels = results['trainingLabels']
    eval_data = results['evalData']
    eval_labels = results['evalLabels']

    res = input('Press 1 to train, press 2 to eval')

    if res == '1':
        train_model()
    elif res == '2':
        eval_model()
    else:
        raise Exception('Option not implemented')


def train_model(training_data: np.ndarray, training_labels: np.ndarray):
    print('Initializing training data')

    # Now we have to train SVM!



def eval_model():
    print('Initializing eval data')



