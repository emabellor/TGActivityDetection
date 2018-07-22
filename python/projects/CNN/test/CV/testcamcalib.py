"""
Script to calibrate camera
"""

import cv2
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import numpy as np
import tkinter as tk
import json
import os

image = None  # type:np.ndarray
image_points = []
object_points = []
do_homo = False


def mouse_callback(event, x_image, y_image, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        master = Tk()
        tk.Label(master, text="X").grid(row=0)
        tk.Label(master, text="Y").grid(row=1)

        e1 = tk.Entry(master)
        e2 = tk.Entry(master)

        e1.grid(row=0, column=1)
        e2.grid(row=1, column=1)

        def add_points():
            global do_homo
            global image

            image_points.append({'x': x_image, 'y': y_image})

            # Draw point
            print('Drawing point')
            radius = 5
            red = (0, 0, 255)
            cv2.rectangle(image, (x_image - radius, y_image - radius), (x_image + radius, y_image + radius),
                          red, cv2.FILLED)
            cv2.imshow('image', image)

            print('Adding points')
            x_object = int(e1.get())
            y_object = int(e2.get())
            object_points.append({'x': x_object, 'y': y_object})

            if len(object_points) == 4:
                do_homo = True

            master.quit()

        tk.Button(master, text='Quit', command=master.quit).grid(row=3, column=0, pady=4)
        tk.Button(master, text='OK', command=add_points).grid(row=3, column=1, pady=4)

        master.mainloop()

        # Destroying window
        master.destroy()


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
            calc_homo(cam_number)


def calc_homo(cam_number: str):
    global image_points
    global object_points

    print('Training')
    print(image_points)
    print(object_points)
    image_points_list = []
    for point in image_points:
        image_points_list.append([point['x'], point['y']])

    object_points_list = []
    for point in object_points:
        object_points_list.append([point['x'], point['y']])

    print('Getting homography')
    matrix, mask = cv2.findHomography(np.array(image_points_list), np.array(object_points_list))

    print('Concatenating into array')
    elem = {'homographyMat': matrix.tolist(), 'imagePoints': image_points, 'objectPoints': object_points}

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
