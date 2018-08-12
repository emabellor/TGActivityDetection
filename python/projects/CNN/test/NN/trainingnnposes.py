import os
from classopenpose import ClassOpenPose
import numpy as np
import cv2
from classutils import ClassUtils
from classdescriptors import ClassDescriptors
from classnn import ClassNN


def main():
    print('Initializing main function')

    # Initializing instances
    instance_pose = ClassOpenPose()

    folder_training_1 = '/home/mauricio/Poses/Walk_Front'
    folder_training_2 = '/home/mauricio/Poses/Vehicle'
    folder_training_3 = '/home/mauricio/Poses/Tires'

    data_1 = get_sets_folder(folder_training_1, 0, instance_pose)
    data_2 = get_sets_folder(folder_training_2, 1, instance_pose)
    data_3 = get_sets_folder(folder_training_3, 2, instance_pose)

    data_training = data_1[0] + data_2[0] + data_3[0]
    label_training = data_1[1] + data_2[1] + data_3[1]

    data_eval = data_1[2] + data_2[2] + data_3[2]
    label_eval = data_1[3] + data_2[3] + data_3[3]






def get_sets_folder(base_folder, label, instance_pose: ClassOpenPose):
    list_files = os.listdir(base_folder)
    features_array = None

    # Work 70, 30
    total_files = len(list_files)
    training = int(total_files * 70 / 100)

    tr_features = list()
    tr_labels = list()

    eval_features = list()
    eval_labels = list()

    for index, file in enumerate(list_files):
        full_name = os.path.join(base_folder, file)

        image = cv2.imread(full_name)

        if image is None:
            raise Exception('Error reading image: {0}'.format(full_name))

        arr = instance_pose.recognize_image(image)

        if len(arr) != 1:
            raise Exception('Invalid len for image {0}: {1}'.format(full_name, len(arr)))

        person_array = arr[0]
        integrity = ClassUtils.check_point_integrity(person_array)

        if not integrity:
            raise Exception('Invalid integrity for points in image: {0}'.format(full_name))

        results = ClassDescriptors.get_person_descriptors(person_array)

        if index < training:
            tr_features.append(results['angles'])
            tr_labels.append(label)
        else:
            eval_features.append(results['angles'])
            eval_labels.append(label)

    return tr_features, tr_labels, eval_features, eval_labels


if __name__ == '__main__':
    main()
