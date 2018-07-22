"""
Script to test calibrate camera
"""

from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os.path
import cv2
import numpy as np
import json
from classutils import ClassUtils


config = None  # type: str


def main():
    global config

    print('Initializing main function')
    text = input('Insert camera number: ')

    base_dir = '/home/mauricio/Oviedo/CameraCalibration/' + text + '/calibration.json'
    print('Reading ' + base_dir)

    if not os.path.isfile(base_dir):
        print('File does not exists')
    else:
        with open(base_dir, 'r') as content_file:
            config = content_file.read()

        print('Opening file')
        Tk().withdraw()

        print('Generating elements in list')

        init_dir = '/home/mauricio/Oviedo/CameraCalibration/' + text
        options = {'initialdir': init_dir}
        filename = askopenfilename(**options)

        if not filename:
            print('File not selected')
        else:
            print('Loading file ' + filename)

            image = cv2.imread(filename)

            if image is None:
                print('Cant read image ' + filename)
            else:
                show_image(image)


def show_image(image: np.ndarray):
    global config

    print('Drawing points')
    dict_config = json.loads(config)

    image_points = dict_config['imagePoints']
    for point in image_points:
        x = int(point['x'])
        y = int(point['y'])

        radius = 5
        red = (0, 0, 255)
        cv2.rectangle(image, (x - radius, y - radius), (x + radius, y + radius), red, cv2.FILLED)

    print('Showing image')

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)
    cv2.imshow('image', image)
    print('Press any key to exit')
    cv2.waitKey(0)


def mouse_callback(event, x_image, y_image, flags, param):
    global config

    if event == cv2.EVENT_LBUTTONDOWN:
        print('Event click - Evaluating elems ' + str(x_image) + ' ' + str(y_image))
        dict_config = json.loads(config)

        homo_mat = np.asarray(dict_config['homographyMat'], dtype='float')
        point = np.float32([x_image, y_image, 1])

        result = ClassUtils.project_points(homo_mat, point)

        print(result)
        print('Done')


if __name__ == '__main__':
    main()
