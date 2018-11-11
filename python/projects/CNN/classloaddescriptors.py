import random
import os
from classutils import ClassUtils
import json
import numpy as np
from enum import Enum
import math


class EnumDesc(Enum):
    ANGLES = 0
    ANGLES_TRANSFORMED = 1
    POINTS = 2
    ALL = 3
    ALL_TRANSFORMED = 4


class ClassLoadDescriptors:

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
        ('/home/mauricio/Pictures/PosesNew/Squat_Right', 0.05, 7),
        ('/home/mauricio/Pictures/PosesNew/Extend_Left', 0.05, 8),
        ('/home/mauricio/Pictures/PosesNew/Extend_Right', 0.05, 9),
    ]

    seed = 1234

    @classmethod
    def load_file_descriptors(cls, full_path: str,  type_desc: EnumDesc):
        with open(full_path, 'r') as text_file:
            arr_json = text_file.read()

        params = json.loads(arr_json)
        angles = params['angles']
        transformed_points = params['transformedPoints']

        # Fill training and eval list
        # Use angles and position information
        data_to_add = cls._get_descriptor_list(angles, transformed_points, type_desc)

        data_np = np.asanyarray(data_to_add, np.float)
        return data_np

    @classmethod
    def load_pose_descriptors(cls, type_desc: EnumDesc):
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

            for folder_data in cls.list_folder_data:
                if folder_data[2] == classes_number:
                    classes_number += 1
                    cont = True
                    break

        # Iterate folder
        for index, item in enumerate(cls.list_folder_data):
            folder = item[0]
            min_score = item[1]
            label = item[2]

            list_files = os.listdir(folder)
            random.Random(cls.seed).shuffle(list_files)

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
                data_to_add = cls._get_descriptor_list(angles, transformed_points, type_desc)
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
        for folder, _, label in cls.list_folder_data:
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

        results = {
            'trainingData': training_data_np,
            'trainingLabels': training_labels_np,
            'evalData': eval_data_np,
            'evalLabels': eval_labels_np,
            'trainingFiles': training_files_np,
            'evalFiles': eval_files_np,
            'labelNames': label_names,
            'classesNumber': classes_number
        }

        return results

    @classmethod
    def _get_descriptor_list(cls, angles, transformed_points, type_desc: EnumDesc):
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