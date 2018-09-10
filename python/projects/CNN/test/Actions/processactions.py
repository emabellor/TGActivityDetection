from tkinter import Tk
from tkinter import filedialog
from classopenpose import ClassOpenPose
from classnn import ClassNN
from classutils import ClassUtils
from classdescriptors import ClassDescriptors
import os
import cv2
import numpy as np

min_pose_score = 0.05
model_dir = '/home/mauricio/models/nn_classifier'


def main():
    print('Initializing main function')

    # Initializing instances
    instance_pose = ClassOpenPose()
    instance_net = ClassNN.load_from_params(model_dir)

    # Withdrawing list
    Tk().withdraw()

    # Select directory to process
    init_dir = '/home/mauricio/CNN/Images'
    options = {'initialdir': init_dir}
    dir_name = filedialog.askdirectory(**options)

    if not dir_name:
        print('Directory not selected')
    else:
        # Loading images
        list_files = os.listdir(dir_name)
        list_files.sort()

        desc_list = list()

        for file in list_files:
            full_path = os.path.join(dir_name, file)

            print('Processing image {0}'.format(full_path))
            image = cv2.imread(full_path)
            arr = instance_pose.recognize_image(image)

            arr_pass = list()
            for person_arr in arr:
                if ClassUtils.check_vector_integrity_part(person_arr, min_pose_score):
                    arr_pass.append(person_arr)

            if len(arr_pass) != 1:
                print('Invalid len {0} for image {1}'.format(len(arr_pass), full_path))
                continue
            else:
                result_des = ClassDescriptors.get_person_descriptors(arr_pass[0], min_pose_score)
                descriptor_arr = result_des['fullDesc']

                # Add descriptors to list
                desc_list.append(descriptor_arr)

        # Convert to numpy array
        print('Total poses: {0}'.format(len(desc_list)))

        # Transform list and predict
        desc_list_np = np.asarray(desc_list, dtype=np.float)
        print('ndim pose list: {0}'.format(desc_list_np.ndim))

        list_classes = list()
        predict_results = instance_net.predict_model_array(desc_list_np)
        for result in predict_results:
            list_classes.append(result['classes'])

        print('Predict results: {0}'.format(list_classes))
        print('Classes label: {0}'.format(instance_net.label_names))

        print('Done!')


if __name__ == '__main__':
    main()
