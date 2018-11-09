from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import json
import numpy as np
from classutils import ClassUtils
from classdescriptors import ClassDescriptors
import math

threshold_angle = 45
threshold_rp = 100


def main():
    print('Initializing main function')

    # Withdrawing tkinter interface
    Tk().withdraw()

    # Loading elements from list
    init_dir = ClassUtils.activity_base_path

    options = {
        'initialdir': init_dir,
        'filetypes': (("JSON Files", "*.json"),
                      ("All files", "*.*"))
    }
    filename = askopenfilename(**options)

    if filename is None:
        raise Exception('Filename not selected!!!')

    print('Filename selected: {0}'.format(filename))

    # Redrawing trajectory
    # Re-scale trajectory to known points
    min_x = -900
    max_x = 1200

    min_y = -900
    max_y = 1200

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    width_plane = 800
    height_plane = 800

    img_plane = np.zeros((height_plane, width_plane, 3), np.uint8)

    # Blank image
    img_plane[:, :] = (255, 255, 255)

    # Opening filename
    with open(filename, 'r') as f:
        json_txt = f.read()

    json_data = json.loads(json_txt)

    # Iterating over data selection - list poses
    list_poses = json_data["listPoses"]
    list_rp, list_action_poses = ClassDescriptors.get_moving_action_poses(list_poses)

    # Only for drawing!
    for i in range(1, len(list_poses)):
        pose1 = list_poses[i - 1]
        pose2 = list_poses[i]

        # Extracting position from pose
        global_pos1 = pose1['globalPosition']
        global_pos2 = pose2['globalPosition']

        # Transforming points
        pt_plane1 = transform_point(global_pos1, min_x, min_y, width_plane, height_plane, delta_x, delta_y)
        pt_plane2 = transform_point(global_pos2, min_x, min_y, width_plane, height_plane, delta_x, delta_y)

        # Draw line using OpenCV library
        thickness = 3
        cv2.line(img_plane, pt_plane1, pt_plane2, (0, 0, 0), thickness)

        # Now that works, detect loitering using Ko Method
        # Link: http://ijssst.info/Vol-15/No-2/data/3251a254.pdf

        # Calculating distance
        if i == 0:
            rect_rad = 6
            # Draw RP point into image
            pt1 = pt_plane1[0] - rect_rad, pt_plane1[1] - rect_rad
            pt2 = pt_plane1[0] + rect_rad, pt_plane1[1] + rect_rad
            cv2.rectangle(img_plane, pt1, pt2, (0, 0, 255), -1)

        # Check if index is in point_rp
        is_rp = False
        for rp in list_rp:
            if 'index' not in rp:
                print('Hello!')

            if rp['index'] == i:
                is_rp = True
                break

        if is_rp:
            rect_rad = 6
            pt1 = pt_plane2[0] - rect_rad, pt_plane2[1] - rect_rad
            pt2 = pt_plane2[0] + rect_rad, pt_plane2[1] + rect_rad
            cv2.rectangle(img_plane, pt1, pt2, (0, 0, 255), -1)
        else:
            rect_rad = 3
            pt1 = pt_plane2[0] - rect_rad, pt_plane2[1] - rect_rad
            pt2 = pt_plane2[0] + rect_rad, pt_plane2[1] + rect_rad
            cv2.rectangle(img_plane, pt1, pt2, (0, 255, 0), -1)

    print('Total trajectories: {0}'.format(len(list_action_poses)))

    # Showing image result
    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)

    cv2.imshow('main_window', img_plane)
    print('Image Loaded! Press a key to continue!')
    cv2.waitKey()

    print('Done')


def transform_point(point, min_x, min_y, width_plane, height_plane, delta_x, delta_y):
    # Draw line into list
    pt_plane_x = int((point[0] - min_x) * width_plane / delta_x)
    pt_plane_y = height_plane - int((point[1] - min_y) * height_plane / delta_y)
    return pt_plane_x, pt_plane_y


if __name__ == '__main__':
    main()
