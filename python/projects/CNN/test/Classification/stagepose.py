from classutils import ClassUtils
from classnn import ClassNN
import os
import json
import numpy as np
from enum import Enum
from classsvm import ClassSVM


class Option(Enum):
    NN = 1
    SVM = 2


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


def main():
    print('Initializing Main Function')

    # Initialize classifier instance - NN
    classes_number = 10
    hidden_number = 60
    learning_rate = 0.04
    nn_classifier = ClassNN(model_dir=ClassNN.model_dir_pose,
                            classes=classes_number,
                            hidden_number=hidden_number,
                            learning_rate=learning_rate)

    # Initialize classifier instance - SVM
    svm_classifier = ClassSVM(ClassSVM.path_model_pose)

    res = input('Press 1 to train using NN - 2 to train using SVM - 3 to train all: ')
    if res == '1':
        calculate_poses(Option.NN, nn_classifier, svm_classifier)
    elif res == '2':
        calculate_poses(Option.SVM, nn_classifier, svm_classifier)
    elif res == '3':
        # Train all
        calculate_poses(Option.NN, nn_classifier, svm_classifier)
        calculate_poses(Option.SVM, nn_classifier, svm_classifier)
    else:
        raise Exception('Option not recognized: {0}'.format(res))


def calculate_poses(option: Option, nn_classifier: ClassNN, svm_classifier: ClassSVM):
    print('Calculating poses using nn')

    # Recalculate all poses and get confidence
    for classInfo in list_classes:
        folder = classInfo['folderPath']
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                ext = ClassUtils.get_filename_extension(full_path)

                if '_rawdata' in file and ext == '.json':
                    print('Processing file: {0}'.format(full_path))

                    with open(full_path, 'r') as f:
                        file_txt = f.read()

                    file_json = json.loads(file_txt)
                    list_poses = file_json['listPoses']

                    for pose in list_poses:
                        angles = pose['angles']
                        transformed_points = pose['transformedPoints']

                        # Using transformed points and angles
                        list_desc = list()
                        list_desc += ClassUtils.get_flat_list(transformed_points)

                        # Convert to numpy
                        list_desc_np = np.asanyarray(list_desc, np.float)

                        if option == Option.NN:
                            result = nn_classifier.predict_model_fast(list_desc_np)
                        else:
                            result = svm_classifier.predict_model(list_desc_np)

                        pose['class'] = int(result['classes'])
                        pose['probabilities'] = result['probabilities'].tolist()

                    # Writing again into file
                    file_txt = json.dumps(file_json, indent=4)
                    new_full_path = ClassUtils.change_ext_training(full_path,
                                                                   '{0}_posedata'.format(option.value))

                    with open(new_full_path, 'w') as f:
                        f.write(file_txt)

                    # Done

    print('Done processing elements')


if __name__ == '__main__':
    main()
