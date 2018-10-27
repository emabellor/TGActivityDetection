from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import json
import numpy as np
from classutils import ClassUtils
import math

threshold_angle = 45
threshold_rp = 100

def main():
    print('Initializing main function')

    # Withdrawing tkinter interface
    Tk().withdraw()

    # Loading elements from list
    init_dir = '/home/mauricio/Pictures/CNN/Images'

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
    min_x = -200
    max_x = 1000

    min_y = -900
    max_y = 1200

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    width_plane = 1000
    height_plane = 500

    img_plane = np.zeros((height_plane, width_plane, 3), np.uint8)

    # Blank image
    img_plane[:, :] = (255, 255, 255)

    # Opening filename
    with open(filename, 'r') as f:
        json_txt = f.read()

    json_data = json.loads(json_txt)

    # Iterating over data selection - list poses
    list_poses = json_data["listPoses"]

    list_rp = list()
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
        rect_rad = 6
        if len(list_rp) == 0:
            # Draw RP point into image
            pt1 = pt_plane1[0] - rect_rad, pt_plane1[1] - rect_rad
            pt2 = pt_plane1[0] + rect_rad, pt_plane1[1] + rect_rad
            cv2.rectangle(img_plane, pt1, pt2, (0, 0, 255), -1)

            list_rp.append(global_pos1)

        # Get last RP
        point_rp = list_rp[-1]

        dis = ClassUtils.get_euclidean_distance_pt(point_rp, global_pos2)
        if dis > threshold_rp:
            pt1 = pt_plane2[0] - rect_rad, pt_plane2[1] - rect_rad
            pt2 = pt_plane2[0] + rect_rad, pt_plane2[1] + rect_rad
            cv2.rectangle(img_plane, pt1, pt2, (0, 0, 255), -1)

            list_rp.append(global_pos2)

    # Calculate angle between rp points
    print('Total rp points: {0}'.format(len(list_rp)))
    count_variations = 0
    for i in range(2, len(list_rp)):
        point0 = list_rp[i - 2]
        point1 = list_rp[i - 1]
        point2 = list_rp[i]

        angle_points = ClassUtils.get_angle(point0, point1, point2)

        angle_deg = angle_points * 180 / math.pi
        angle_change = 180 - angle_deg

        if angle_change > threshold_angle:
            count_variations += 1

    print('Trajectory changes: {0}'.format(count_variations))

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
