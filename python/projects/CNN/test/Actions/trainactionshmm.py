import os
from classnn import ClassNN
from classhmm import ClassHMM
from classopenpose import ClassOpenPose
from classutils import ClassUtils
from classdescriptors import ClassDescriptors
import cv2
import numpy as np
from tkinter import Tk
from tkinter import filedialog
import json


nn_model_dir = '/home/mauricio/models/nn_classifier'
hnn_model_folder = '/home/mauricio/models/hmm'
min_score = 0.05
hidden_states = 6


def main():
    print('Initializing main function')

    # Withdrawing tkinter

    # Loading model dirs
    list_folder_data = [
        ('/home/mauricio/CNN/Classes/Door', 0.05),
        ('/home/mauricio/CNN/Classes/Tires', 0.05),
        ('/home/mauricio/CNN/Classes/Walk', 0.05),
    ]

    list_hmm = []

    for folder_data in list_folder_data:
        label_name = get_label_from_folder(folder_data[0])
        full_model_dir = os.path.join(hnn_model_folder, '{0}.pkl'.format(label_name))
        list_hmm.append(ClassHMM(full_model_dir))

    # Initializing instances
    instance_pose = ClassOpenPose()
    instance_nn = ClassNN.load_from_params(nn_model_dir)

    option = input('Select 1 to train, 2 to eval hmm, 3 to preprocess: ')

    if option == '1':
        print('Train hmm selection')
        train_hmm(list_folder_data, list_hmm, instance_nn, instance_pose)
    elif option == '2':
        eval_hmm(list_folder_data, list_hmm, instance_nn, instance_pose)
    elif option == '3':
        recalculate = False
        pre_process_images(instance_pose, list_folder_data, recalculate)
    else:
        print('Invalid argument: {0}'.format(option))


def get_label_from_folder(folder: str):
    names = folder.split('/')
    label_name = names[-1]
    return label_name


def train_hmm(list_folder_data:list, list_hmm: list, instance_nn: ClassNN, instance_pose: ClassOpenPose):
    for index, folder_data in enumerate(list_folder_data):
        label_name = get_label_from_folder(folder_data[0])
        print('Training class {0}'.format(label_name))

        sub_folders = [f.path for f in os.scandir(folder_data[0]) if f.is_dir()]
        list_desc = []

        for folder in sub_folders:
            list_desc.append(get_poses_seq(folder, instance_nn, instance_pose, only_json=True))

        print('Total desc: {0}'.format(len(list_desc)))

        list_hmm[index].train(list_desc, hidden_states)
        print('Trained class {0}'.format(label_name))


def eval_hmm(list_folder_data:list, list_hmm: list, instance_nn: ClassNN, instance_pose: ClassOpenPose):
    # Ask folder using tkinter
    init_dir = '/home/mauricio/CNN/Classes/'
    options = {'initialdir': init_dir}
    dir_name = filedialog.askdirectory(**options)

    if not dir_name:
        print('Directory not selected')
    else:
        desc = get_poses_seq(dir_name, instance_nn, instance_pose, only_json=True)
        desc_np = np.asarray(desc, dtype=np.int)

        # Evaluating score in all list
        score = 0
        hmm_winner = -1

        for index, hmm_model in enumerate(list_hmm):
            hmm_score = hmm_model.get_score(desc_np)

            if hmm_winner == -1:
                score = hmm_score
                hmm_winner = index
            elif hmm_score > score:
                score = hmm_score
                hmm_winner = index

        if hmm_winner == -1:
            raise Exception('Invalid hmm winner')

        label_name = get_label_from_folder(list_folder_data[hmm_winner][0])
        print('Prediction label: {0} score: {1}', label_name, score)


def get_poses_seq(folder: str, instance_nn: ClassNN, instance_pose: ClassOpenPose, only_json=False):
    # List all folders
    list_files = []
    for file in os.listdir(folder):
        list_files.append(os.path.join(folder, file))

    # Sorting elements
    list_files.sort()

    # Get elements
    list_desc = list()
    for path in list_files:
        ext = ClassUtils.get_filename_extension(path)
        if only_json:
            if ext != '.json':
                print('Ignoring file: {0}'.format(path))
                continue

            with open(path, 'r') as file:
                person_arr_str = file.read()
                person_arr = json.loads(person_arr_str)
        else:
            if ext != '.jpg':
                print('Ignoring file: {0}'.format(path))
                continue

            print('Processing file {0}'.format(path))
            image = cv2.imread(path)

            arr = instance_pose.recognize_image(image)

            arr_pass = []
            for person_arr in arr:
                if ClassUtils.check_vector_integrity_part(person_arr, min_score):
                    arr_pass.append(person_arr)

            if len(arr_pass) != 1:
                print('Ignoring file {0} - Len arr_pass: {1}'.format(path, len(arr_pass)))
                continue

            person_arr = arr_pass[0]

        result_desc = ClassDescriptors.get_person_descriptors(person_arr, min_score)
        list_desc.append(result_desc['fullDesc'])

    list_desc_np = np.asarray(list_desc, np.float)
    results = instance_nn.predict_model_array(list_desc_np)

    list_classes = []
    for result in results:
        list_classes.append(result['classes'])

    return list_classes


def pre_process_images(instance_pose: ClassOpenPose, list_folders_scores: list, recalculate: bool):
    print('Start pre_processing images')

    all_folders = []
    for folder_data in list_folders_scores:
        sub_folders = [f.path for f in os.scandir(folder_data[0]) if f.is_dir()]

        for sub_folder in sub_folders:
            all_folders.append(sub_folder)

    for folder in all_folders:
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


if __name__ == '__main__':
    main()
