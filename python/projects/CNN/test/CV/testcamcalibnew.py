"""
Script to calibrate camera
The points are saved in default positions
When calibrated, the user must be set the position and the rotation angle of the camera
The rotation angle must be clockwise - degrees
"""

import cv2
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import numpy as np
import tkinter as tk
import json
import os

image = None  # type:np.ndarray
image_points = list()
object_points = list()
do_homo = False
center = []
ok_params = False
angle_deg = 0

# Default positions
# Change if model of cells changes
default_positions = [
    [-125, 750],
    [125, 750],
    [-125, 250],
    [125, 250]
]


def mouse_callback(event, x_image, y_image, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global do_homo
        global image
        global default_positions

        image_points.append([x_image, y_image])

        # Draw point
        print('Drawing point')
        radius = 5
        red = (0, 0, 255)
        cv2.rectangle(image, (x_image - radius, y_image - radius), (x_image + radius, y_image + radius),
                      red, cv2.FILLED)
        cv2.imshow('image', image)

        # Add object point
        index = len(image_points) - 1
        x_object = default_positions[index][0]
        y_object = default_positions[index][1]
        object_point = [x_object, y_object]
        object_points.append(object_point)
        print('Object point loaded: {0}'.format(object_point))

        if len(object_points) == 4:
            do_homo = True


def main():
    global image
    global do_homo
    global image_points
    global object_points

    print('OpenCV calibration')

    Tk().withdraw()
    cam_number = input('Insert camera number: ')

    base_dir = '/home/mauricio/Oviedo/CameraCalibration/' + cam_number + '/calibration.json'

    global continue_load
    continue_load = True
    print(base_dir)

    if os.path.exists(base_dir):
        question = Tk()

        def no_click():
            global continue_load
            continue_load = False
            question.quit()

        tk.Label(question, text="File exists. Do you want to overwrite?").grid(row=0)
        tk.Button(question, text='No', command=no_click).grid(row=1, column=1)
        tk.Button(question, text='Yes', command=question.quit).grid(row=1, column=0)

        question.mainloop()
        question.destroy()

    if not continue_load:
        print('Quiting')
    else:
        loading_image(cam_number)


def loading_image(cam_number: str):
    global image
    print('Loading image')
    init_dir = '/home/mauricio/Oviedo/CameraCalibration/' + cam_number
    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if not filename:
        print('File not selected')
    else:
        image = cv2.imread(filename)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', mouse_callback)

        print('Press q to exit!')
        cv2.imshow('image', image)
        while True:
            k = cv2.waitKey(100)
            if do_homo:
                break
            else:
                if k == 113:  # q Key
                    print('Function canceled')
                    break

        if not do_homo:
            print('Canceled')
        else:
            ask_position_angle(cam_number)


def ask_position_angle(cam_number: str):
    master = Tk()
    tk.Label(master, text="x center").grid(row=0)
    tk.Label(master, text="y center").grid(row=1)
    tk.Label(master, text="angle deg").grid(row=2)

    e1 = tk.Entry(master)
    e2 = tk.Entry(master)
    e3 = tk.Entry(master)

    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)
    e3.grid(row=2, column=1)

    def read_params():
        global ok_params
        global center
        global angle_deg

        x_object = int(e1.get())
        y_object = int(e2.get())
        center = [x_object, y_object]

        angle_deg = int(e3.get())
        ok_params = True
        master.quit()

    tk.Button(master, text='Quit', command=master.quit).grid(row=3, column=0, pady=4)
    tk.Button(master, text='OK', command=read_params).grid(row=3, column=1, pady=4)

    master.mainloop()

    # Destroying window
    master.destroy()

    if ok_params:
        calc_homo(cam_number)
    else:
        print('Quitting')


def calc_homo(cam_number: str):
    global image_points
    global object_points
    global center
    global angle_deg

    print('Training')
    print(image_points)
    print(object_points)

    print('Getting homography')
    matrix, mask = cv2.findHomography(np.array(image_points), np.array(object_points))

    print('Concatenating into array')
    elem = {
                'camNumber': cam_number,
                'homographyMat': matrix.tolist(),
                'imagePoints': image_points,
                'objectPoints': object_points,
                'centerPoints': center,
                'angleDegrees': angle_deg
            }

    elem_str = json.dumps(elem)
    print(elem_str)

    print('Writing in file')
    base_dir = '/home/mauricio/Oviedo/CameraCalibration/' + cam_number + '/calibration.json'

    dir_path = os.path.dirname(base_dir)

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    with open(base_dir, 'w') as file:
        file.write(elem_str)

    print('Done!')


if __name__ == '__main__':
    main()
