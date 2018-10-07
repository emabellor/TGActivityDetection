"""
Initializing main function
Loading elements from list
"""

from classutils import ClassUtils
import os
import cv2
from classopenpose import ClassOpenPose
from classdescriptors import ClassDescriptors
import json
from classnn import ClassNN
import numpy as np
import random
from tkinter import Tk
from enum import Enum
import math

seed = 1234

csv_dir = '/home/mauricio/Pictures/PosesNew/data.csv'
csv_train = '/home/mauricio/Pictures/PosesNew/training.csv'
csv_eval = '/home/mauricio/Pictures/PosesNew/eval.csv'
csv_dir_files = '/home/mauricio/Pictures/PosesNew/files.csv'


class EnumDesc(Enum):
    ANGLES = 0
    ANGLES_TRANSFORMED = 1
    POINTS = 2
    ALL = 3
    ALL_TRANSFORMED = 4


def main():
    Tk().withdraw()

    print('Initializing pose classification')

    # Loading list images
    # Folders have indexes
    # Each folder has its own score!

    # This numbers must be maintained
    # Poses 6 and 7 are used for position adjustment
    list_folder_data = [
        ('/home/mauricio/Pictures/PosesNew/Back', 0.05, 0),
        ('/home/mauricio/Pictures/PosesNew/Hands_Left', 0.05, 1),
        ('/home/mauricio/Pictures/PosesNew/Hands_Right', 0.05, 2),
        ('/home/mauricio/Pictures/PosesNew/Front', 0.05, 3),
        ('/home/mauricio/Pictures/PosesNew/Left', 0.05, 4),
        ('/home/mauricio/Pictures/PosesNew/Right', 0.05, 5),
        ('/home/mauricio/Pictures/PosesNew/Squat_Left', 0.05, 6),
        ('/home/mauricio/Pictures/PosesNew/Squat_Right', 0.05, 7)
        # ('/home/mauricio/Pictures/PosesNew/Chat_Left', 0.05, 8),
        # ('/home/mauricio/Pictures/PosesNew/Chat_Right', 0.05, 9),
        # ('/home/mauricio/Pictures/PosesNew/Extend_Left', 0.05, 10),
        # ('/home/mauricio/Pictures/PosesNew/Extend_Right', 0.05, 11),
    ]

    # Delete re calculate option - Enter directly to classify option
    res = input('Press 1 to classify images, '
                '2 to reprocess images, '
                '3 to get transformed points: ')

    if res == '1':
        res = input('Press 1 to classify images using angles, '
                    '2 to classify images using angles transformed, '
                    '3 to classify images using points, '
                    '4 to classify images using all points, '
                    '5 to classify images using all points transformed: ')

        if res == '1' or res == '2':
            # No pose variant descriptors

            list_folder_data = [
                ('/home/mauricio/Pictures/PosesNew/Back', 0.05, 0),
                ('/home/mauricio/Pictures/PosesNew/Hands_Left', 0.05, 1),
                ('/home/mauricio/Pictures/PosesNew/Hands_Right', 0.05, 1),
                ('/home/mauricio/Pictures/PosesNew/Front', 0.05, 0),
                ('/home/mauricio/Pictures/PosesNew/Left', 0.05, 2),
                ('/home/mauricio/Pictures/PosesNew/Right', 0.05, 2),
                ('/home/mauricio/Pictures/PosesNew/Squat_Left', 0.05, 3),
                ('/home/mauricio/Pictures/PosesNew/Squat_Right', 0.05, 3)
            ]

            if res == '1':
                classify_images(list_folder_data, EnumDesc.ANGLES)
            else:
                classify_images(list_folder_data, EnumDesc.ANGLES_TRANSFORMED)
        elif res == '3':
            classify_images(list_folder_data, EnumDesc.POINTS)
        elif res == '4':
            classify_images(list_folder_data, EnumDesc.ALL)
        elif res == '5':
            classify_images(list_folder_data, EnumDesc.ALL_TRANSFORMED)
        else:
            raise Exception('Option not recognized: {0}'.format(res))
    elif res == '2':
        reprocess_images(list_folder_data)
    elif res == '3':
        get_transformed_points()
    else:
        raise Exception('Option not recognized: {0}'.format(res))


# Deprecated
def delete_prev_json_files(list_folders_scores):
    print('Deleting previous json files')
    for folder, _ in list_folders_scores:
        print('Processing folder {0}'.format(folder))

        for file in os.listdir(folder):
            full_path = os.path.join(folder, file)
            ext = ClassUtils.get_filename_extension(full_path)

            if ext == '.json':
                print('Deleting file {0}'.format(full_path))
                os.remove(full_path)

    print('Done deleting files!')


# Deprecated
def pre_process_images(list_folders_scores, recalculate):
    print('Start pre_processing images')

    # Loading instances
    instance_pose = ClassOpenPose()

    for folder, min_score in list_folders_scores:
        for file in os.listdir(folder):
            full_path = os.path.join(folder, file)
            extension = ClassUtils.get_filename_extension(full_path)

            if extension != '.jpg':
                print('Ignoring file {0}'.format(full_path))
                continue

            file_no_ext = ClassUtils.get_filename_no_extension(full_path)
            arr_file_name = os.path.join(folder, '{0}.json'.format(file_no_ext))

            # If image recalculation
            if not recalculate:
                if os.path.isfile(arr_file_name):
                    print('File already processed {0}'.format(full_path))
                    continue

            # Processing file
            print('Processing file {0}'.format(full_path))
            image = cv2.imread(full_path)

            arr, img_draw = instance_pose.recognize_image_tuple(image)

            arr_pass = list()

            # Checking vector integrity for all elements
            # Verify there is at least one arm and one leg
            for elem in arr:
                if ClassUtils.check_vector_integrity_part(elem, min_score):
                    arr_pass.append(elem)

            # If there is more than one person with vector integrity
            if len(arr_pass) != 1:
                for elem in arr_pass:
                    pt1, pt2 = ClassUtils.get_rectangle_bounds(elem, ClassUtils.MIN_POSE_SCORE)
                    cv2.rectangle(img_draw, pt1, pt2, (0, 0, 255), 3)

                cv2.namedWindow('main_window')
                cv2.imshow('main_window', img_draw)
                print(arr)
                print(arr_pass)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                raise Exception('Invalid len: {0} file {1}'.format(len(arr_pass), full_path))

            person_arr = arr_pass[0]

            arr_str = json.dumps(person_arr.tolist())

            with open(arr_file_name, 'w') as text_file:
                text_file.write(arr_str)

    print('Done!')


def classify_images(list_folder_data: list, type_desc: EnumDesc):
    training_data = list()
    training_labels = list()
    training_files = list()
    eval_data = list()
    eval_labels = list()
    eval_files = list()

    classes_number = 0

    cont = True
    while cont:
        cont = False

        for folder_data in list_folder_data:
            if folder_data[2] == classes_number:
                classes_number += 1
                cont = True
                break

    hidden_number = 40
    learning_rate = 0.04
    steps = 20000

    # Initialize classifier instance
    nn_classifier = ClassNN(model_dir=ClassNN.model_dir_pose,
                            classes=classes_number,
                            hidden_number=hidden_number,
                            learning_rate=learning_rate)

    # Iterate folder
    for index, item in enumerate(list_folder_data):
        folder = item[0]
        min_score = item[1]
        label = item[2]

        list_files = os.listdir(folder)
        random.Random(seed).shuffle(list_files)

        total_train = int(len(list_files)) * 70 / 100

        for num_file, file in enumerate(list_files):
            full_path = os.path.join(folder, file)

            extension = ClassUtils.get_filename_extension(full_path)
            if extension != '.json':
                print('Ignoring file {0}'.format(full_path))
                continue

            with open(full_path, 'r') as text_file:
                arr_json = text_file.read()

            params = json.loads(arr_json)
            vectors = params['vectors']
            angles = params['angles']
            transformed_points = params['transformedPoints']

            valid = ClassUtils.check_vector_integrity_pos(vectors, min_score)
            only_pos = ClassUtils.check_vector_only_pos(vectors, min_score)

            if not valid:
                raise Exception('Vector integrity not valid for file: {0}'.format(full_path))

            if only_pos:
                raise Exception('Invalid vector to perform detection')

            # Fill training and eval list
            # Use angles and position information
            data_to_add = get_descriptor_list(angles, transformed_points, type_desc)
            if num_file < total_train:
                training_data.append(data_to_add)
                training_labels.append(label)
                training_files.append(full_path)
            else:
                eval_data.append(data_to_add)
                eval_labels.append(label)
                eval_files.append(full_path)

    # Convert data to numpy array
    training_data_np = np.asanyarray(training_data, dtype=np.float)
    training_labels_np = np.asanyarray(training_labels, dtype=int)

    eval_data_np = np.asanyarray(eval_data, dtype=np.float)
    eval_labels_np = np.asanyarray(eval_labels, dtype=int)

    training_files_np = np.asanyarray(training_files, dtype=np.str)
    eval_files_np = np.asanyarray(eval_files, dtype=np.str)

    # Getting label_names
    label_names = []
    for folder, _, label in list_folder_data:
        names = folder.split('/')
        label_name = names[-1]

        # Check if last character is /
        if len(label_name) == 0:
            label_names = names[-2]

        label_names.append((label_name, label))

    print('Total training: {0}'.format(len(training_labels)))
    print('Total eval: {0}'.format(len(eval_labels)))
    print('Shape training: {0}'.format(training_data_np.shape))
    print('Shape eval: {0}'.format(eval_data_np.shape))
    print('Classes number: {0}'.format(classes_number))

    # Prompt for user input
    selection = input('Training selected {0}. '
                      'Press 1 to train, 2 to evaluate, 3 to predict, 4 to save csv, '
                      '5 to get confusion matrix: '.format(type_desc))

    if selection == '1':
        # Training
        nn_classifier.train_model(train_data=training_data_np,
                                  train_labels=training_labels_np,
                                  label_names=label_names,
                                  steps=steps)

        # Evaluate after training
        nn_classifier.eval_model(eval_data_np, eval_labels_np)
    elif selection == '2':
        # Evaluate
        nn_classifier.eval_model(eval_data_np, eval_labels_np)
    elif selection == '3':
        # Predict
        # Select data to eval
        data_eval = eval_data_np[0]
        label_eval = eval_labels_np[0]

        results = nn_classifier.predict_model(data_eval)
        print('Predict data np: {0}'.format(results))
        print('Expected data np: {0}'.format(label_eval))
    elif selection == '4':
        # Saving file in csv
        total_data = np.concatenate((training_data_np, eval_data_np), axis=0)
        total_labels = np.concatenate((training_labels_np, eval_labels_np), axis=0)
        total_files = np.concatenate((training_files_np, eval_files_np))

        # Add new axis to allow concatenation
        total_labels = total_labels[:, np.newaxis]
        total_files = total_files[:, np.newaxis]

        total_np = np.concatenate((total_data, total_labels), axis=1)

        print('Saving data to CSV in file {0}'.format(csv_dir))
        np.savetxt(csv_dir, total_np, delimiter=',', fmt='%10.10f')
        np.savetxt(csv_dir_files, total_files, delimiter=',', fmt='%s')

        print('Saving training data')
        # Concatenate with new axis
        total_np_train = np.concatenate((training_data_np, training_labels_np[:, np.newaxis]), axis=1)
        total_np_eval = np.concatenate((eval_data_np, eval_labels_np[:, np.newaxis]), axis=1)

        # Saving
        np.savetxt(csv_train, total_np_train, delimiter=',', fmt='%10.10f')
        np.savetxt(csv_eval, total_np_eval, delimiter=',', fmt='%10.10f')

        print('Done writing file in CSV')
    elif selection == '5':
        print('Getting confusion matrix')

        confusion_np = np.zeros((classes_number, classes_number))
        for i in range(eval_data_np.shape[0]):
            data = eval_data_np[i]
            expected = eval_labels_np[i]
            obtained = nn_classifier.predict_model(data)
            class_prediction = obtained['classes']
            print('Class: {0}'.format(class_prediction))

            confusion_np[expected, class_prediction] += 1

        print('Confusion matrix')
        print(confusion_np)
        print('Labels: {0}'.format(label_names))
    else:
        raise Exception('Option not supported')


def get_descriptor_list(angles, transformed_points, type_desc: EnumDesc):
    if type_desc == EnumDesc.ANGLES:
        return angles
    elif type_desc == EnumDesc.POINTS:
        return ClassUtils.get_flat_list(transformed_points)
    elif type_desc == EnumDesc.ALL:
        data_to_add = list()
        data_to_add += angles
        data_to_add += ClassUtils.get_flat_list(transformed_points)
        return data_to_add
    elif type_desc == EnumDesc.ANGLES_TRANSFORMED:
        data_to_add = list()

        for angle in angles:
            sin_angle = math.sin(angle)
            cos_angle = math.cos(angle)

            data_to_add.append(sin_angle)
            data_to_add.append(cos_angle)

        return data_to_add
    elif type_desc == EnumDesc.ALL_TRANSFORMED:
        data_to_add = list()

        for angle in angles:
            sin_angle = math.sin(angle)
            cos_angle = math.cos(angle)

            data_to_add.append(sin_angle)
            data_to_add.append(cos_angle)

        data_to_add += ClassUtils.get_flat_list(transformed_points)
        return data_to_add
    else:
        raise Exception('Invalid option: {0}'.format(type_desc))


def reprocess_images(list_folder_data):
    print('Init reprocess_images')

    for index, item in enumerate(list_folder_data):
        folder = item[0]
        min_score = item[1]

        print('Processing folder: {0}'.format(folder))
        list_files = os.listdir(folder)
        random.Random(seed).shuffle(list_files)

        for num_file, filename in enumerate(list_files):
            file = os.path.join(folder, filename)
            extension = ClassUtils.get_filename_extension(file)

            if extension == '.json':
                with open(file, 'r') as f:
                    data_str = f.read()

                data_json = json.loads(data_str)
                if 'vectors' in data_json:
                    print('Processing json file with new format: {0}'.format(file))
                    person_arr = data_json['vectors']
                else:
                    print('Processing json file: {0}'.format(file))
                    person_arr = data_json

                valid = ClassUtils.check_vector_integrity_pos(person_arr, min_score)
                only_pos = ClassUtils.check_vector_only_pos(person_arr, min_score)

                if not valid:
                    raise Exception('Vector integrity not valid for file: {0}'.format(file))

                if only_pos:
                    raise Exception('Invalid vector to perform detection')

                descriptors = ClassDescriptors.get_person_descriptors(person_arr, min_score, cam_number=0,
                                                                      image=None, calib_params=None,
                                                                      decode_img=False,
                                                                      instance_nn_pose=None)

                with open(file, 'w') as f:
                    f.write(json.dumps(descriptors))

                transformed_points = descriptors['transformedPoints']

                # Save pose for debugging purposes
                re_scale_factor = 100
                new_points = ClassDescriptors.re_scale_pose_factor(transformed_points,
                                                                   re_scale_factor, min_score)
                img_pose = ClassDescriptors.draw_pose_image(new_points, min_score, is_transformed=True)
                new_file_name = ClassUtils.get_filename_no_extension(file) + '_1.jpg'
                cv2.imwrite(new_file_name, img_pose)

    print('Done!')


def get_transformed_points():
    min_score = 0.05
    instance_pose = ClassOpenPose()
    res = ClassDescriptors.load_images_comparision_ext(instance_pose, min_score, load_one_img=True)

    person_arr = res['pose1']
    descriptors = ClassDescriptors.get_person_descriptors(person_arr, min_score, cam_number=0,
                                                          image=None, calib_params=None,
                                                          decode_img=False,
                                                          instance_nn_pose=None)
    transformed_points = descriptors['transformedPoints']
    print(transformed_points)

    # Save pose for debugging purposes
    re_scale_factor = 100
    new_points = ClassDescriptors.re_scale_pose_factor(transformed_points,
                                                       re_scale_factor, min_score)
    img_pose = ClassDescriptors.draw_pose_image(new_points, min_score, is_transformed=True)

    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('main_window', img_pose)

    cv2.waitKey()
    cv2.destroyAllWindows()

    print('Done!')


if __name__ == '__main__':
    main()
