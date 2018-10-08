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
classes = 0


def main():
    global classes

    list_folder_data = [
        {
            'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Door'),
            'label': 0
        },
        {
            'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Moving'),
            'label': 1
        },
        {
            'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Squat'),
            'label': 2
        }
    ]

    suffix = input('Select image index: Ej: _a -> angles image, _p -> points image, _s -> poses image, '
                   '_b -> angles image no resize: ')

    classes = len(list_folder_data)
    instance_train = ClassCNN(ClassCNN.model_dir_action, classes, img_width, img_height, depth)
    classify_images(list_folder_data, instance_train, suffix)


def classify_images(list_folder_data: list, instance_train: ClassCNN, suffix: str):
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

                if suffix in full_path:
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

    if training_data_np.shape[0] == 0:
        raise Exception('No files found for suffix: {0}'.format(suffix))

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
    instance_train.update_batch_size(training_data_np.shape[0])
    instance_train.train_model(training_data_np, training_labels_np)

    # Done!
    print('Done!')


def eval_model(eval_data_np: np.ndarray, eval_labels_np: np.ndarray, instance_train: ClassCNN):
    # Evaluate
    instance_train.eval_model(eval_data_np, eval_labels_np)

    # Getting confussion matrix
    print('Getting confusion matrix')

    confusion_np = np.zeros((classes, classes))
    for i in range(eval_data_np.shape[0]):
        data = eval_data_np[i]
        expected = eval_labels_np[i]
        obtained = instance_train.predict_model_fast(data)
        class_prediction = obtained['classes']
        print('Class: {0}'.format(class_prediction))

        confusion_np[expected, class_prediction] += 1

    print('Confusion matrix')
    print(confusion_np)

    print('Done!')


if __name__ == '__main__':
    main()
