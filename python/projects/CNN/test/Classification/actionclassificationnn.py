from classnn import ClassNN
import os
from classutils import ClassUtils
import random
import json
import time
import numpy as np
from enum import Enum

hidden_number = 100
poses_number = 10
samples_size = 28
seed = 1234
pose_classes = 10
pose_hidden_number = 60


class Desc(Enum):
    POSES = 1,
    ALL = 2,
    POINTS = 3


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
    },
    {
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Quiet'),
        'label': 3
    },
    {
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Plumbs'),
        'label': 4
    }
]


def main():
    print('Initializing main function')

    # Initializing instances
    classes = len(list_folder_data)
    instance_nn_train = ClassNN(ClassNN.model_dir_action, classes, hidden_number)
    instance_nn_pose = ClassNN(ClassNN.model_dir_pose, pose_classes, pose_hidden_number)

    res = input('Press 1 to work with poses, press 2 to work with all - press 3 to work with points ')

    if res == '1':
        load_descriptors(instance_nn_train, instance_nn_pose, Desc.POSES)
    elif res == '2':
        load_descriptors(instance_nn_train, instance_nn_pose, Desc.ALL)
    elif res == '3':
        load_descriptors(instance_nn_train, instance_nn_pose, Desc.POINTS)
    else:
        print('Option not recognized: {0}'.format(res))


def load_descriptors(instance_nn_train: ClassNN, instance_nn_pose: ClassNN, pose_type: Desc):
    training_data = list()
    training_labels = list()
    eval_data = list()
    eval_labels = list()
    training_files = list()
    eval_files = list()

    for index, item in enumerate(list_folder_data):
        folder = item['folderPath']
        label = item['label']

        print('Processing folder path: {0}'.format(folder))

        num_file = 0
        list_paths = list()
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                extension = ClassUtils.get_filename_extension(full_path)

                if extension == '.json':
                    list_paths.append(full_path)

        total_samples = len(list_paths)
        total_train = int(total_samples * 80 / 100)

        # Shuffle samples
        random.Random(seed).shuffle(list_paths)

        for full_path in list_paths:
            # Reading data
            with open(full_path, 'r') as f:
                json_txt = f.read()

            json_data = json.loads(json_txt)

            list_poses = json_data['listPoses']

            # Sampling data
            descriptor = list()
            for index_size in range(samples_size):
                index_pose = int(len(list_poses) * index_size / samples_size)
                pose = list_poses[index_pose]
                transformed_points = pose['transformedPoints']
                angles = pose['angles']

                list_desc = list()
                list_desc += angles
                list_desc += ClassUtils.get_flat_list(transformed_points)

                if pose_type == Desc.POSES:
                    list_desc_np = np.asanyarray(list_desc, dtype=np.float)
                    res = instance_nn_pose.predict_model_fast(list_desc_np)

                    # Add descriptor with probabilities
                    for elem in res['probabilities']:
                        descriptor.append(elem)
                elif pose_type == Desc.ALL:
                    for elem in list_desc:
                        descriptor.append(elem)
                elif pose_type == Desc.POINTS:
                    list_flat = ClassUtils.get_flat_list(transformed_points)
                    for elem in list_flat:
                        descriptor.append(elem)
                else:
                    raise Exception('Pose type not recognized: {0}'.format(pose_type))

            if num_file < total_train:
                training_data.append(descriptor)
                training_labels.append(label)
                training_files.append(full_path)
            else:
                eval_data.append(descriptor)
                eval_labels.append(label)
                eval_files.append(full_path)

            num_file += 1

    # Convert data to numpy array
    training_data_np = np.asanyarray(training_data, dtype=np.float)
    training_labels_np = np.asanyarray(training_labels, dtype=int)

    eval_data_np = np.asanyarray(eval_data, dtype=np.float)
    eval_labels_np = np.asanyarray(eval_labels, dtype=int)

    print('Shape images training: {0}'.format(training_data_np.shape))
    print('Shape labels training: {0}'.format(training_labels_np.shape))

    if training_data_np.shape[0] == 0:
        raise Exception('No files found!')

    res = input('Press 1 to train - 2 to eval: ')

    if res == '1':
        train_model(training_data_np, training_labels_np, eval_data_np, eval_labels_np, instance_nn_train, steps=30000)
    elif res == '2':
        eval_model(eval_data_np, eval_labels_np, instance_nn_train)
    else:
        raise Exception('Option not implemented!')


def train_model(train_data_np: np.ndarray, train_labels_np: np.ndarray,
                eval_data_np: np.ndarray, eval_labels_np: np.ndarray, instance_nn_train: ClassNN, steps):

    print('Training model into list')

    # Init training!
    # instance_train.update_batch_size(training_data_np.shape[0])
    start = time.time()
    instance_nn_train.train_model(train_data_np, train_labels_np, steps=steps)
    end = time.time()

    # Performing data evaluation
    eval_model(eval_data_np, eval_labels_np, instance_nn_train)


def eval_model(eval_data_np: np.ndarray, eval_labels_np: np.ndarray, instance_nn_train: ClassNN):
    classes = len(list_folder_data)

    # Evaluate
    instance_nn_train.eval_model(eval_data_np, eval_labels_np)

    # Getting confussion matrix
    print('Getting confusion matrix')

    confusion_np = np.zeros((classes, classes))
    for i in range(eval_data_np.shape[0]):
        data = eval_data_np[i]
        expected = eval_labels_np[i]
        obtained = instance_nn_train.predict_model_fast(data)
        class_prediction = obtained['classes']

        confusion_np[expected, class_prediction] += 1

    print('Confusion matrix')
    print(confusion_np)

    print('Done!')


if __name__ == '__main__':
    main()
