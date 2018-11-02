import os
from classutils import ClassUtils
import random
from classcnn import ClassCNN
import cv2
import numpy as np
import time

seed = 1234
img_width = 28
img_height = 28
depth = 3
suffix = ''
list_models = list()
steps = 15000

list_folder_data = [
    {
        # Cls 0
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Door'),
    },
    {
        # Cls1
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Squat'),
    },
    {
        # Cls 2
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Quiet'),
    },
    {
        # Cls 3
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Plumbs'),
    }
]

list_folder_no_data = [
    {
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Moving')
    }
]


def main():
    global suffix
    global list_models
    print('Initializing main function')

    suffix = input('Select image index: Ej: _a -> angles image, _p -> points image, _s -> poses image no resize '
                   '_b -> angles image no resize, _o -> points no resize, _r -> action rem position: ')

    # One vs all classification!
    classes = 2

    # Create CNN for each class
    for cls in range(len(list_folder_data)):
        dir_model = os.path.join(ClassCNN.model_dir_ac_one_all, str(cls))
        print('Path model for class {0}: {1}'.format(cls, dir_model))
        instance_train = ClassCNN(dir_model, classes, img_width, img_height, depth, batch_size=32)
        list_models.append(instance_train)

    classify_images()


def classify_images():
    global suffix
    global list_folder_data
    global list_models

    training_data = list()
    training_labels = list()
    training_files = list()
    eval_data = list()
    eval_labels = list()
    eval_files = list()

    # Adding normal training data!
    for cls, item in enumerate(list_folder_data):
        folder = item['folderPath']

        num_file = 0
        list_paths = list()
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)

                if suffix in full_path and '.bmp' in full_path:
                    list_paths.append(full_path)

        total_samples = len(list_paths)
        total_train = int(total_samples * 80 / 100)

        # Shuffle samples
        random.Random(seed).shuffle(list_paths)

        for full_path in list_paths:
            img_cv = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

            if img_cv.shape != (img_height, img_width):
                img_cv = cv2.resize(img_cv, (img_height, img_width))

            img_cv = np.resize(img_cv, (img_height, img_width, 1))

            if num_file < total_train:
                training_data.append(img_cv)
                training_labels.append(cls)
                training_files.append(full_path)
            else:
                eval_data.append(img_cv)
                eval_labels.append(cls)
                eval_files.append(full_path)

            num_file += 1

    # Adding no training data!
    for item in list_folder_no_data:
        folder = item['folderPath']
        num_file = 0
        list_paths = list()
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)

                if suffix in full_path and '.bmp' in full_path:
                    list_paths.append(full_path)

        total_samples = len(list_paths)
        total_train = int(total_samples * 80 / 100)

        # Shuffle samples
        random.Random(seed).shuffle(list_paths)

        for full_path in list_paths:
            img_cv = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

            if img_cv.shape != (img_height, img_width):
                img_cv = cv2.resize(img_cv, (img_height, img_width))

            img_cv = np.resize(img_cv, (img_height, img_width, 1))

            if num_file < total_train:
                training_data.append(img_cv)
                training_labels.append(-1)
                training_files.append(full_path)
            else:
                eval_data.append(img_cv)
                eval_labels.append(-1)
                eval_files.append(full_path)
            num_file += 1

    print('Total training data: {0}'.format(len(training_data)))
    print('Total eval data: {0}'.format(len(eval_data)))

    res = input('Press 1 to train, 2 to eval individual, 3 to eval global: ')

    if res == '1':
        train_set(training_data, training_labels, eval_data, eval_labels)
    elif res == '2':
        for cls in range(len(list_folder_data)):
            eval_set_cls(cls, eval_data, eval_labels)
    elif res == '3':
        eval_set_global(eval_data, eval_labels)
    else:
        raise Exception('Invalid option: {0}'.format(res))


def train_set(training_data: list, training_labels: list, eval_data: list, eval_labels: list):
    global list_folder_data
    global list_models

    print('Initializing train set')
    for cls, data in enumerate(list_folder_data):
        instance_train = list_models[cls]
        training_data_cls = list()
        training_labels_cls = list()

        for i in range(len(training_data)):
            training_data_cls.append(training_data[i])

            if training_labels[i] == cls:
                training_labels_cls.append(1)
            else:
                training_labels_cls.append(0)

        # Converting to numpy
        training_data_np = np.asanyarray(training_data_cls, dtype=np.float)
        training_labels_np = np.asanyarray(training_labels_cls, dtype=int)

        print('Training model {0}'.format(cls))
        start = time.time()
        instance_train.train_model(training_data_np, training_labels_np, steps=steps)
        end = time.time()

        print('Elapsed training: {0}'.format(end - start))

        eval_set_cls(cls, eval_data, eval_labels)


def eval_set_cls(cls, eval_data: list, eval_labels: list):
    global list_folder_data
    global list_models

    eval_data_cls = list()
    eval_labels_cls = list()

    for i in range(len(eval_data)):
        eval_data_cls.append(eval_data[i])

        if eval_labels[i] == cls:
            eval_labels_cls.append(1)
        else:
            eval_labels_cls.append(0)

    eval_data_np = np.asanyarray(eval_data_cls, dtype=np.float)
    eval_labels_np = np.asanyarray(eval_labels_cls, dtype=np.int)

    print('Evaluating model for class: {0}'.format(cls))

    instance_train = list_models[cls]
    instance_train.eval_model(eval_data_np, eval_labels_np)

    print('Done!')


def eval_set_global(eval_data: list, eval_labels: list):
    global list_models
    print('Evaluating set global into list')

    # Iterating over elements to get list
    count = 0
    classes = len(list_models) + 1
    confusion_np = np.zeros((classes, classes))
    for index, data in enumerate(eval_data):
        max_prob = 0
        selected_cls = -1
        thresh_prob = 0.7
        label = eval_labels[index]

        for cls in range(len(list_models)):
            model = list_models[cls]
            data_np = np.asanyarray(data, dtype=np.float)

            result = model.predict_model_fast(data_np)
            predict_cls = result['classes']
            probability = result['probabilities'][predict_cls]
            print('Class: {0} - Label: {3} - Result: {2} - Probabilities: {1}'.format(cls, probability,
                                                                                      predict_cls, label))

            if probability > max_prob and probability > thresh_prob and predict_cls == 1:
                selected_cls = cls
                max_prob = probability

        if selected_cls == label:
            count += 1

        expected = label
        if expected == -1:
            expected = len(list_models)

        class_prediction = selected_cls
        if class_prediction == -1:
            class_prediction = len(list_models)

        confusion_np[expected, class_prediction] += 1

    precision = count / len(eval_labels)
    print('Model precision: {0}'.format(precision))

    print('Printing confusion matrix')
    print(confusion_np)

    print('Done!')


if __name__ == '__main__':
    main()
