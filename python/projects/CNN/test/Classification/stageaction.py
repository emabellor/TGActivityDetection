import os
from classutils import ClassUtils
import random
import json
import numpy as np
from classnn import ClassNN
from classcnn import ClassCNN
from enum import Enum
import cv2

seed = 1234


class Option(Enum):
    NN = 1
    CNN = 2
    HMM = 3


cnn_image_width = 28
cnn_image_height = 28
depth = 1

list_classes = [
    {
        # Cls 0
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Door'),
    },
    {
        # Cls 1
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Down'),
    },
    {
        # Cls 2
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Loitering'),
    },
    {
        # Cls 3
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Plumbs'),
    },
    {
        # Cls 4
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Squat'),
    },
    {
        # Cls 5
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Up'),
    },
    {
        # Cls 6
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Walk'),
    }
]


list_classes_classify = [
    {
        # Cls 0 - Door
        'folderPaths': [
            os.path.join(ClassUtils.activity_base_path, 'Door')
        ],
    },
    {
        # Cls 1 - Plumbs
        'folderPaths': [os.path.join(ClassUtils.activity_base_path, 'Plumbs')]
    },
    {
        # Cls 2 - Squat
        'folderPaths': [os.path.join(ClassUtils.activity_base_path, 'Squat')]
    },
    {
        # Cls 3 - N/A
        'folderPaths': [os.path.join(ClassUtils.activity_base_path, 'Walk'),
                        os.path.join(ClassUtils.activity_base_path, 'Loitering')]
    }
]

# Cls 4 - 7: Same with zone
# Cls 8: Moving - Init and finish out of zone
# Cls 9: Moving - Init in zone, finish out of zone
# Cls 10: Moving - Init out of zone, finish in zone
# Cls 11: Moving - Init and finish in zone


def main():
    print('Initializing main function')

    res = input('Press 1 to train NN - 2 to train CNN - 3 to train HMM - 4 to Train All: ')
    if res == '1':
        load_and_train(Option.NN)
    elif res == '2':
        load_and_train(Option.CNN)
    elif res == '3':
        load_and_train(Option.HMM)
    elif res == '4':
        load_and_train(Option.NN)
        load_and_train(Option.CNN)
        load_and_train(Option.HMM)
    else:
        raise Exception('Option not recognized: {0}'.format(res))


def load_and_train(option: Option):
    # Dataset is in form of _x_partialdata.json
    # Iterate until there is no more elems
    base_data = 1
    while True:
        # Loading elements into list
        training_list_poses = list()
        training_labels = list()
        eval_list_poses = list()
        eval_labels = list()

        for index, item in enumerate(list_classes_classify):
            folders = item['folderPaths']
            label = index

            list_list_poses = list()
            for folder in folders:
                for root, _, files in os.walk(folder):
                    for file in files:
                        full_path = os.path.join(root, file)
                        extension = ClassUtils.get_filename_extension(full_path)

                        if extension == '.json' and '_{0}_partialdata'.format(base_data) in full_path:
                            with open(full_path, 'r') as f:
                                json_txt = f.read()

                            json_data = json.loads(json_txt)

                            for pose_action in json_data['listPosesAction']:
                                # Take all pose action
                                if not pose_action['moving']:
                                    list_list_poses.append(pose_action['listPoses'])

            total_samples = len(list_list_poses)
            if total_samples == 0:
                # There is no data for base_cls - Breaking
                break

            total_train = int(total_samples * 80 / 100)
            print('Total samples: {0}'.format(total_samples))

            # Shuffle samples
            random.Random(seed).shuffle(list_list_poses)

            num_file = 0
            for list_poses in list_list_poses:
                if num_file < total_train:
                    training_list_poses.append(list_poses)
                    training_labels.append(label)
                else:
                    eval_list_poses.append(list_poses)
                    eval_labels.append(label)

                num_file += 1

        if option == Option.NN:
            print('Train nn for base data: {0}'.format(base_data))
            train_nn_cnn(training_list_poses, training_labels, eval_list_poses, eval_labels, option)
        elif option == Option.CNN:
            print('Train cnn for base data: {0}'.format(base_data))
            train_nn_cnn(training_list_poses, training_labels, eval_list_poses, eval_labels, option)
        elif option == Option.HMM:
            print('Train hmm for base data: {0}'.format(base_data))
            train_hmm(training_list_poses, training_labels, eval_list_poses, eval_labels)
        else:
            raise Exception('Option not recognized')


def train_nn_cnn(training_list_poses, training_labels, eval_list_poses, eval_labels, option):
    print('Init NN Training')

    if option != Option.NN and option != option.CNN:
        raise Exception('Option not valid: {0}'.format(option))

    # Training labels
    list_descriptors = list()
    for index_pose in range(len(training_list_poses)):
        list_poses = training_list_poses[index_pose]

        if option == Option.NN:
            descriptor = get_nn_descriptor(list_poses)
        else:
            descriptor = get_cnn_descriptor(list_poses)

        list_descriptors.append(descriptor)

    training_descriptors_np = np.asanyarray(list_descriptors, dtype=np.float)
    training_labels_np = np.asanyarray(training_labels, dtype=np.int)

    # Eval labels
    list_descriptors = list()
    for index_pose in range(len(eval_list_poses)):
        list_poses = eval_list_poses[index_pose]

        if option == Option.NN:
            descriptor = get_nn_descriptor(list_poses)
        else:
            descriptor = get_cnn_descriptor(list_poses)

        list_descriptors.append(descriptor)

    eval_descriptors_np = np.asanyarray(list_descriptors, dtype=np.float)
    eval_labels_np = np.asanyarray(eval_labels, dtype=np.int)

    # Initializing training instance
    classes = len(list_classes_classify)

    if option == Option.NN:
        hidden_number = 50
        instance_model = ClassNN(ClassNN.model_dir_action, classes, hidden_number)
    else:
        instance_model = ClassCNN(ClassCNN.model_dir_action, classes, cnn_image_height, cnn_image_height,
                                  depth, batch_size=32)

    print('Training nn model')
    instance_model.train_model(training_descriptors_np, training_labels_np)

    print('Model trained - Evaluating')
    instance_model.eval_model(eval_descriptors_np, eval_labels_np)

    # Evaluating all elements
    for folder_info in list_classes:
        folder = folder_info['folderPath']
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                if '_partialdata' in full_path:
                    process_file_nn_cnn(full_path, instance_model, option)


def get_nn_descriptor(list_poses):
    samples_size = 28

    descriptor = list()
    for index_size in range(samples_size):
        index_pose = int(len(list_poses) * index_size / samples_size)
        pose = list_poses[index_pose]

        for elem in pose['probabilities']:
            descriptor.append(elem)

    return descriptor


def get_cnn_descriptor(list_poses):
    samples_size = 28

    max_confidence = 1

    image_height = cnn_image_height
    image_width = len(list_poses)

    image_np = np.zeros((image_height, image_width), dtype=np.uint8)

    for index_size in range(samples_size):
        index_pose = int(len(list_poses) * index_size / samples_size)
        pose = list_poses[index_pose]

        for index, value in enumerate(pose['probabilities']):
            pixel_value = int(value * 255 / max_confidence)
            image_np[index, index_pose] = pixel_value

    image_res = cv2.resize(image_np, (cnn_image_height, cnn_image_width))
    image_res = np.resize(image_res, (cnn_image_height, cnn_image_width, 1))
    return image_res


def process_file_nn_cnn(full_path, instance_model, option: Option):
    with open(full_path, 'r') as f:
        json_txt = f.read()

    json_data = json.loads(json_txt)
    list_actions = list()

    list_list_poses = json_data['listPosesAction']
    for list_poses_data in list_list_poses:
        list_poses = list_poses_data['listPoses']
        moving = list_poses_data['moving']
        in_zone = list_poses_data['inZone']
        in_zone_before = list_poses_data['inZoneBefore']

        # Descriptor not moving
        if not moving:
            descriptor = get_nn_descriptor(list_poses)
            descriptor_nn = np.asanyarray(descriptor, dtype=np.float)

            res = instance_model.predict_model_fast(descriptor_nn)
            cls = int(res['classes'])

            if in_zone:
                cls += 4

            probabilities = res['probabilities'].tolist()
            list_prob = list()

            if in_zone:
                for i in range(4):
                    list_prob.append(0)
                list_prob += probabilities
            else:
                list_prob += probabilities
                for i in range(4):
                    list_prob.append(0)

            for i in range(4):
                list_prob.append(0)

            data_action = {
                'class': cls,
                'probabilities': list_prob,
                'count': len(list_poses)
            }
        else:
            if not in_zone and not in_zone_before:
                cls = 8
            elif in_zone and not in_zone_before:
                cls = 9
            elif not in_zone and in_zone_before:
                cls = 10
            else:
                cls = 11

            probabilities = list()
            for i in range(12):
                if i == cls:
                    probabilities.append(1)
                else:
                    probabilities.append(0)

            data_action = {
                'class': cls,
                'probabilities': probabilities,
                'count': len(list_poses)
            }

        list_actions.append(data_action)

    json_data = {
        'listActions': list_actions
    }

    # Saving full action info into list
    json_txt = json.dumps(json_data, indent=4)
    new_file_path = ClassUtils.change_ext_training(full_path, '{0}_actiondata'.format(option.value))

    with open(new_file_path, 'w') as f:
        f.write(json_txt)

    # Done!


def train_hmm(training_list_poses, training_labels, eval_list_poses, eval_labels):
    print('Train HMM')


if __name__ == '__main__':
    main()



