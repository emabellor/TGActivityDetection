from classutils import ClassUtils
from classnn import ClassNN
import os
import json
import numpy as np


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

    res = input('Press 1 to train using NN: ')
    if res == '1':
        calculate_poses_nn()


def calculate_poses_nn():
    print('Calculating poses using nn')

    classes_number = 10
    hidden_number = 60
    learning_rate = 0.04

    # Initialize classifier instance
    nn_classifier = ClassNN(model_dir=ClassNN.model_dir_pose,
                            classes=classes_number,
                            hidden_number=hidden_number,
                            learning_rate=learning_rate)

    # Recalculate all poses and get confidence
    for classInfo in list_classes:
        folder = classInfo['folderPath']
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                ext = ClassUtils.get_filename_extension(full_path)

                if ext == '.json' and '_rawdata' in file:
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
                        list_desc += angles
                        list_desc += ClassUtils.get_flat_list(transformed_points)

                        # Convert to numpy
                        list_desc_np = np.asanyarray(list_desc, np.float)
                        result = nn_classifier.predict_model_fast(list_desc_np)

                        pose['class'] = int(result['classes'])
                        pose['probabilities'] = result['probabilities'].tolist()

                    # Writing again into file
                    file_txt = json.dumps(file_json, indent=4)
                    new_full_path = ClassUtils.change_ext_training(full_path, 'posedata')

                    with open(new_full_path, 'w') as f:
                        f.write(file_txt)

                    # Done

    print('Done processing elements')


if __name__ == '__main__':
    main()
