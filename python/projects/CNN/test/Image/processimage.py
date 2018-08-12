from tkinter.filedialog import askopenfilename
from tkinter import Tk
import cv2
import os
from classopenpose import ClassOpenPose
from classdescriptors import ClassDescriptors
from classmjpegconverter import ClassMjpegConverter
from classutils import ClassUtils
import math


def main():
    # Boilerplate code
    Tk().withdraw()

    print('Initializing process image')

    print('Loading instances')
    instance_pose = ClassOpenPose()

    print('Opening filename')

    init_dir = '/home/mauricio/Pictures'
    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if not filename:
        filename = '/home/mauricio/Pictures/Poses/left_walk/636550788328999936_420.jpg'

    print('Reading video extension')
    extension = os.path.splitext(filename)[1]

    if extension != '.jpg' and extension != '.jpeg':
        raise Exception('Extension is not jpg or jpeg')
    else:
        print('Opening filename')

        image = cv2.imread(filename)
        print('Loading image and array')

        arr, processed_img = instance_pose.recognize_image_tuple(image)

        print('Showing image to generate elems')
        cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)

        arr_pass = list()
        min_score = 0.05

        # Checking vector integrity for all elements
        # Verify there is at least one arm and one leg
        for elem in arr:
            if ClassUtils.check_vector_integrity_part(elem, min_score):
                arr_pass.append(elem)

        if len(arr_pass) != 1:
            print('There is more than one person in the image')
            cv2.imshow('main_window', processed_img)
            cv2.waitKey(0)
        else:
            person_array = arr_pass[0]
            print('Person array: {0}'.format(person_array))
            generate_descriptors(person_array, processed_img, min_score)


def generate_descriptors(person_array, processed_img, min_score):
    results = ClassDescriptors.get_person_descriptors(person_array, min_score)

    print('Results: {0}'.format(results))

    cv2.imshow('main_window', processed_img)
    cv2.waitKey(0)

    print('Done!')


if __name__ == '__main__':
    main()
