import os
from classopenpose import ClassOpenPose
import numpy as np
import cv2
import shutil
from classutils import ClassUtils
from classdescriptors import ClassDescriptors
from classnn import ClassNN


def main():
    print('Initializing main function')

    # Initializing instances
    instance_pose = ClassOpenPose()

    folder_training_1 = '/home/mauricio/Pictures/Poses/Walk_Front'
    folder_training_2 = '/home/mauricio/Pictures/Poses/Vehicle'
    folder_training_3 = '/home/mauricio/Pictures/Poses/Tires'

    data_1 = get_sets_folder(folder_training_1, 0, instance_pose)
    data_2 = get_sets_folder(folder_training_2, 1, instance_pose)
    data_3 = get_sets_folder(folder_training_3, 1, instance_pose)

    data_training = np.asarray(data_1[0] + data_2[0] + data_3[0], dtype=np.float)
    label_training = np.asarray(data_1[1] + data_2[1] + data_3[1], dtype=np.int)

    data_eval = np.asarray(data_1[2] + data_2[2] + data_3[2], dtype=np.float)
    label_eval = np.asarray(data_1[3] + data_2[3] + data_3[3], dtype=np.int)

    print(data_training)
    print(label_training)

    print('Len data_training: {0}'.format(data_training.shape))
    print('Len data_eval: {0}'.format(data_eval.shape))

    classes = 3
    hidden_neurons = 15
    model_dir = '/tmp/nnposes/'

    instance_nn = ClassNN(model_dir, classes, hidden_neurons)

    option = input('Press 1 to train the model, 2 to eval, otherwise to predict')

    if option == '1':
        print('Training the model')

        # Delete previous folder to avoid conflicts in the training process
        if os.path.isdir(model_dir):
            # Removing directory
            shutil.rmtree(model_dir)

        instance_nn.train_model(data_training, label_training)
    elif option == '2':
        print('Eval the model')
        instance_nn.eval_model(data_eval, label_eval)
    else:
        print('Not implemented!')

    print('Done!')


def get_sets_folder(base_folder, label, instance_pose: ClassOpenPose):
    list_files = os.listdir(base_folder)

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

        print('Processing image: {0}'.format(full_name))
        arr = instance_pose.recognize_image(image)

        # Selecting array for list of vectors
        person_array = []
        if len(arr) == 0:
            raise Exception('Invalid len for image {0}: {1}'.format(full_name, len(arr)))
        elif len(arr) == 1:
            person_array = arr[0]

            integrity = ClassUtils.check_vector_integrity_low(person_array)

            if not integrity:
                raise Exception('Invalid integrity for points in image: {0}'.format(full_name))
        else:
            found = False
            for arr_index in arr:
                integrity = ClassUtils.check_vector_integrity_low(arr_index)

                if integrity:
                    found = True
                    person_array = arr_index
                    break

            if not found:
                raise Exception('Cant find a valid person array for image {0}'.format(full_name))

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
