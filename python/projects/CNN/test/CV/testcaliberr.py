"""
testcamcalib.py
Script to measure error from selecting input using mouse
"""
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from classutils import ClassUtils
import tkinter as tk
import os
import numpy as np

list_points = list()
list_points_item = list()
image = ''
image_path = ''
save_points = False
out_csv_filename = '/home/mauricio/CSV/err_calc.csv'
number_iter = 30


def mouse_callback(event, x_image, y_image, flags, param):
    global save_points
    global image
    global image_path
    global list_points
    global list_points_item
    global number_iter

    if event == cv2.EVENT_LBUTTONDOWN:
        list_points_item.append(x_image)
        list_points_item.append(y_image)

        print('Saving point {0} from iter {1}'.format(len(list_points), len(list_points_item) / 2))

        # Draw point
        print('Drawing point')
        radius = 5
        red = (0, 0, 255)
        cv2.rectangle(image, (x_image - radius, y_image - radius), (x_image + radius, y_image + radius),
                      red, cv2.FILLED)

        cv2.imshow('image', image)

        if len(list_points_item) == 8:
            # Reloading image
            image = cv2.imread(image_path)

            # Saving item list

            list_points.append(list_points_item.copy())

            # Reloading list
            list_points_item.clear()
            print('Saving point')

            cv2.imshow('image', image)

        if len(list_points) == number_iter:
            save_points = True


def main():
    print('Initializing main function')

    option = input('Select 1 to create error csv, 2 to create error propagation: ')

    if option == '1':
        make_error_csv()
    elif option == '2':
        calc_error_prop()
    else:
        print('Invalid selection!')


def make_error_csv():
    # Loading image using Tkinter
    Tk().withdraw()

    # Asking directory
    global list_points
    global save_points
    global image
    global image_path

    print('OpenCV calibration')

    Tk().withdraw()

    print('Loading image')

    init_dir = '/home/mauricio/Pictures/'
    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if not filename:
        print('File not selected')
    else:
        image_path = filename
        image = cv2.imread(filename)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', mouse_callback)

        print('Press q to exit!')
        cv2.imshow('image', image)
        while True:
            k = cv2.waitKey(100)
            if save_points:
                break
            else:
                if k == 113:  # q Key
                    print('Function canceled')
                    break

        if not save_points:
            print('Canceled')
        else:
            save_points_func()


def save_points_func():
    global out_csv_filename
    global list_points

    # Saving points from list
    list_points_np = np.array(list_points, dtype=int)

    header = 'pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y, pt4_x, pt4_y'
    np.savetxt(out_csv_filename, list_points_np, delimiter=",", header=header)

    cv2.destroyAllWindows()
    print('Document saved in {0}'.format(out_csv_filename))
    print('Done')


def calc_error_prop():
    # Calculating error propagation
    # Points obtained from file /home/mauricio/CSV/err_calc.ods

    list_image_points = np.array([
        [171, 96.93],
        [440.3, 91.73],
        [32.76, 453.43],
        [590.23, 447.53]
    ])

    list_obj_points = np.array([
        [-125, 750],
        [125, 750],
        [-125, 250],
        [125, 250]
    ])

    mat, _ = cv2.findHomography(list_image_points, list_obj_points)
    print('Homography matrix: {0}'.format(mat))

    for index in range(list_image_points.shape[0]):
        point_des = 4
        image_point = list_image_points[index]
        image_point[0] += point_des
        image_point[1] += point_des

        print('Image point: {0}'.format(image_point))

        obj_point = list_obj_points[index]
        proj_point = ClassUtils.project_points(mat, image_point)

        delta_x = proj_point[0] - obj_point[0]
        delta_y = proj_point[1] - obj_point[1]

        print('Projected point: {0}'.format(proj_point))
        print('DeltaX: {0} - DeltaY: {1}'.format(delta_x, delta_y))

    print('Done!')


if __name__ == '__main__':
    main()
