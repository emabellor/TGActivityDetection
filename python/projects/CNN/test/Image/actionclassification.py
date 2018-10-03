from classcnn import ClassCNN
import os
import cv2
import numpy as np
import random
from classutils import ClassUtils
import json

seed = 1234

# Loading instances
img_width = 28
img_height = 28
depth = 1


def main():

    list_folder_data = [
        {
            'folderPath': '/home/mauricio/Pictures/CNN/Classes/No_Mov/Door',
            'label': 0
        },
        {
            'folderPath': '/home/mauricio/Pictures/CNN/Classes/No_Mov/Moving',
            'label': 1
        },
        {
            'folderPath': '/home/mauricio/Pictures/CNN/Classes/No_Mov/Squat',
            'label': 2
        }
    ]

    classes = len(list_folder_data)
    instance_train = ClassCNN(ClassCNN.model_dir_action, classes, img_width, img_height, depth,
                              train_steps=2000)
    classify_images(list_folder_data, instance_train)


def classify_images(list_folder_data: list, instance_train: ClassCNN):
    training_data = list()
    training_labels = list()
    training_files = list()
    eval_data = list()
    eval_labels = list()
    eval_files = list()

    for index, item in enumerate(list_folder_data):
        folder = item['folderPath']
        label = item['label']

        total_samples = len(os.listdir(folder))
        total_train = int(total_samples * 80 / 100)

        num_file = 0
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)

                if '_p' in full_path:
                    img_cv = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

                    if img_cv.shape != (img_height, img_width):
                        img_cv = cv2.resize(img_cv, (img_height, img_width))

                    img_cv = np.resize(img_cv, (img_height, img_width, 1))

                    if num_file < total_train:
                        training_data.append(img_cv)
                        training_labels.append(label)
                        training_files.append(full_path)
                    else:
                        eval_data.append(img_cv)
                        eval_labels.append(label)
                        eval_files.append(full_path)

                    num_file += 1

    # Convert data to numpy array
    training_data_np = np.asanyarray(training_data, dtype=np.float)
    training_labels_np = np.asanyarray(training_labels, dtype=int)

    eval_data_np = np.asanyarray(eval_data, dtype=np.float)
    eval_labels_np = np.asanyarray(eval_labels, dtype=int)

    training_files_np = np.asanyarray(training_files, dtype=np.str)
    eval_files_np = np.asanyarray(eval_files, dtype=np.str)

    print('Shape images training: {0}'.format(training_data_np.shape))
    print('Shape labels training: {0}'.format(training_labels_np.shape))

    print('Initializing main function')
    res = input('Press 1 to train - 2 to eval: ')

    if res == '1':
        train_model(training_data_np, training_labels_np, instance_train)
    elif res == '2':
        eval_model(eval_data_np, eval_labels_np, instance_train)
    else:
        raise Exception('Option not implemented!')


def train_model(training_data_np: np.ndarray, training_labels_np: np.ndarray, instance_train: ClassCNN):
    print('Training model into list')
    # Init training!
    instance_train.train_model(training_data_np, training_labels_np)

    # Done!
    print('Done!')


def eval_model(eval_data_np: np.ndarray, eval_labels_np: np.ndarray, instance_train: ClassCNN):
    # Evaluate
    instance_train.eval_model(eval_data_np, eval_labels_np)

    print('Done!')


if __name__ == '__main__':
    main()
