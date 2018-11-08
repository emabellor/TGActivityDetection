"""
positionlabeling.py
Program to classify position from list
Generation from rectangles
"""

import cv2
from classutils import ClassUtils
import numpy as np
import os
from sys import platform
import json

image_width = 800
image_height = 600
min_x = -200
max_x = 1000
min_y = -800
max_y = 800

flag_down = False

point_init_img = (0, 0)
point_end_img = (0, 0)
list_object_points = list()
list_point_rect_obj = list()

# Creating image into list
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
image[:, :] = (255, 255, 255)


def main():
    global list_object_points, image, list_point_rect_obj
    print('Initializing main function')

    list_cams = [419, 420, 421, 428, 429, 430]

    # Loading files
    for cam in list_cams:
        cam_str = str(cam)

        calib_params = ClassUtils.load_cam_calib_params(cam_str)
        object_points = calib_params['objectPoints']

        center_points = calib_params['centerPoints']
        angle_degrees = calib_params['angleDegrees']

        if angle_degrees == 0:
            for point in object_points:
                point[0] += center_points[0]
                point[1] += center_points[1]
        elif angle_degrees == 180:
            for point in object_points:
                point[0] = center_points[0] - point[0]
                point[1] = center_points[1] - point[1]
        else:
            raise Exception('Angle deg not implemented: {0}'.format(angle_degrees))

        print('Object points for cam: {0} - {1}'.format(cam, object_points))
        list_object_points.append({
            'camNumber': cam,
            'objectPoints': object_points
        })

    # Loading Zone Points
    path = '/home/mauricio/Oviedo/CameraCalibration/ZonePoints/calibration.json'
    if platform == 'win32':
        path = 'C:\\SharedFTP\\CameraCalibration\\ZonePoints\\calibration.json'

    if os.path.exists(path):
        with open(path, 'r') as f:
            obj_json = f.read()

        obj_data = json.loads(obj_json)
        list_points_temp = obj_data['listRectanglePoints']

        # Converting to tuple
        for points in list_points_temp:
            list_point_rect_obj.append([(points[0][0], points[0][1]), (points[1][0], points[1][1])])

    draw_image()

    # Showing image
    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('main_window', mouse_callback)

    print('Showing images. Press ESC to exit. Press Q to reset rectangles, Press S to save')
    while True:
        cv2.imshow('main_window', image)
        key = cv2.waitKey(100)

        if key != -1:
            print('Key pressed: {0}'.format(key))

            if key == 27:
                break
            elif key == 113:
                print('Cleaning object list')
                list_point_rect_obj.clear()
                draw_image()
            elif key == 115:
                save_points()
                # Exiting!
                break

    cv2.destroyAllWindows()
    print('Done!')


def save_points():
    global list_point_rect_obj

    print('Saving into list')
    path = ClassUtils.zone_calib_path

    base_dir = os.path.dirname(path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    data_obj = {
        'listRectanglePoints': list_point_rect_obj
    }

    data_json = json.dumps(data_obj)
    with open(path, 'w') as f:
        f.write(data_json)

    print('File written in {0}'.format(path))


def mouse_callback(event, x_image, y_image, flags, param):
    global flag_down, point_init_img, point_end_img, list_point_rect_obj

    if event == cv2.EVENT_LBUTTONDOWN:
        print('Event left button down')
        flag_down = True
        point_init_img = (x_image, y_image)
        point_end_img = (x_image, y_image)
        draw_image()
    elif event == cv2.EVENT_MOUSEMOVE:
        if flag_down:
            point_end_img = (x_image, y_image)
            draw_image()
    elif event == cv2.EVENT_LBUTTONUP:
        print('Event left button up')
        flag_down = False
        point_end_img = (x_image, y_image)

        list_point_rect_obj.append([point_init_img, point_end_img])
        draw_image()


def draw_image():
    global list_object_points, point_init_img, point_end_img, image, list_point_rect_obj

    image[:, :] = (255, 255, 255)

    # Drawing points
    for item in list_object_points:
        cam = item['camNumber']
        object_points = item['objectPoints']

        list_drawn_points = list()
        for point in object_points:
            pos_x = int((point[0] - min_x) * image_width / (max_x - min_x))
            pos_y = int(image_height - (point[1] - min_y) * image_height / (max_y - min_y))

            radius = 3
            cv2.rectangle(image, (pos_x - radius, pos_y - radius), (pos_x + radius, pos_y + radius),
                          (255, 0, 0), -1)
            list_drawn_points.append([pos_x, pos_y])

    # Drawing list points
    for points in list_point_rect_obj:
        cv2.rectangle(image, points[0], points[1], (0, 0, 255), 3)

    if flag_down:
        print('Drawing rectangle points {0} {1}'.format(point_init_img, point_end_img))
        # Drawing rectangle
        cv2.rectangle(image, point_init_img, point_end_img, (0, 0, 255), 3)


if __name__ == '__main__':
    main()
