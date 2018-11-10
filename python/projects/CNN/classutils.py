import numpy as np
import os
import json
from datetime import timedelta
import math
from datetime import datetime
import cv2
import uuid
from colormath.color_objects import AdobeRGBColor, LabColor, HSVColor, sRGBColor
from colormath.color_diff import delta_e_cie2000
from sys import platform
import logging
from os import path

logger = logging.getLogger('ClassUtils')


class ClassUtils:
    # Class Variables
    MIN_POSE_SCORE = 0.1

    video_base_path = '/home/mauricio/Videos/Oviedo'
    cnn_base_path = '/home/mauricio/Pictures/CNN'
    no_img_path = '/home/mauricio/Pictures/novideo.jpg'
    cnn_folder_mov = '/home/mauricio/Mov/Images'
    activity_base_path = '/home/mauricio/Pictures/CNN/ClassesActivity'
    zone_calib_path = '/home/mauricio/Oviedo/CameraCalibration/ZonePoints/calibration.json'

    # Variable changing - If platform is not same
    if platform == 'win32':
        video_base_path = 'C:\\VideosPython'
        cnn_folder_mov = 'C:\\CNN\\Images'
        cnn_base_path = 'C:\\SharedFTP'
        no_img_path = 'C:\\novideo.jpg'
        activity_base_path = 'C:\\SharedFTP\\ClassesActivity'
        zone_calib_path = 'C:\\SharedFTP\\CameraCalibration\\ZonePoints\\calibration.json'

    cnn_folder = os.path.join(cnn_base_path, 'Images/')
    cnn_partial_folder_mov = os.path.join(cnn_base_path, 'Images_Partial/Mov')
    cnn_partial_folder_no_mov = os.path.join(cnn_base_path, 'Images_Partial/No_Mov')
    cnn_class_folder = os.path.join(cnn_base_path, 'Classes')

    @staticmethod
    def project_points(homo_matrix: np.ndarray, point2d: np.ndarray):
        point = np.float32([point2d[0], point2d[1], 1])

        result = np.matmul(homo_matrix, point)

        new_x = result[0] / result[2]
        new_y = result[1] / result[2]

        return np.asanyarray([new_x, new_y], dtype=np.float)

    @classmethod
    def project_points_angle(cls, homo_matrix: np.ndarray, point2d: np.ndarray,
                             point_center: np.ndarray, angle_degrees: int):
        local_projected_point = cls.project_points(homo_matrix, point2d)
        transformed_point = cls.transform_angle_point(local_projected_point, point_center, angle_degrees)
        return transformed_point

    @classmethod
    def load_homo_mat(cls, camera_number: str):
        dict_config = cls.load_cam_calib_params(camera_number)
        homo_mat = np.asarray(dict_config['homographyMat'], dtype='float')

    @staticmethod
    def load_cam_calib_params(camera_number: str):
        base_dir = '/home/mauricio/Oviedo/CameraCalibration/' + camera_number + '/calibration.json'
        if platform == 'win32':
            base_dir = 'C:\\SharedFTP\\CameraCalibration\\' + camera_number + '\\calibration.json'

        if not os.path.exists(base_dir):
            raise Exception(base_dir + ' does not exist in system')
        else:
            with open(base_dir, 'r') as content_file:
                config = content_file.read()

            dict_config = json.loads(config)
            return dict_config

    @staticmethod
    def cam_calib_exists(camera_number: str):
        base_dir = '/home/mauricio/Oviedo/CameraCalibration/' + camera_number + '/calibration.json'
        if platform == 'win32':
            base_dir = 'C:\\SharedFTP\\CameraCalibration\\' + camera_number + '\\calibration.json'

        return os.path.exists(base_dir)

    @staticmethod
    def ticks_to_datetime(ticks: float):
        dt = datetime(1, 1, 1) + timedelta(microseconds=ticks / 10)
        return dt

    @staticmethod
    def datetime_to_ticks(date: datetime):
        microseconds = int((date - datetime(1, 1, 1)).total_seconds() * 1000000 * 10)
        return microseconds

    @classmethod
    def get_euclidean_distance_pt(cls, pt1, pt2):
        if type(pt1) is dict:
            x1 = pt1['x']
            y1 = pt1['y']

        else:
            x1 = pt1[0]
            y1 = pt1[1]

        if type(pt2) is dict:
            x2 = pt2['x']
            y2 = pt2['y']
        else:
            x2 = pt2[0]
            y2 = pt2[1]

        return cls.get_euclidean_distance(x1, y1, x2, y2)

    @staticmethod
    def get_euclidean_distance(x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

    @staticmethod
    def write_bin_to_file(path_file: str, bin_array):
        with open(path_file, 'wb') as newFile:
            newFile.write(bin_array)

    @staticmethod
    # Get angle in radians
    def get_angle(point1, point2, point3):
        # P1 is the base point
        # P2 is where the two lines intersect
        # P3 is the destination point

        # Points must be in anticlockwise order
        # To get more information please refer to this link
        # https://www.geeksforgeeks.org/orientation-3-ordered-points/

        # Integrity of all points must be valid

        if type(point1) is dict:
            x1 = point1['x']
            y1 = point1['y']
        else:
            x1 = point1[0]
            y1 = point1[1]

        if type(point2) is dict:
            x2 = point2['x']
            y2 = point2['y']
        else:
            x2 = point2[0]
            y2 = point2[1]

        if type(point3) is dict:
            x3 = point3['x']
            y3 = point3['y']
        else:
            x3 = point3[0]
            y3 = point3[1]

        dx21 = x2 - x1
        dx31 = x3 - x1
        dx32 = x3 - x2
        dy21 = y2 - y1
        dy31 = y3 - y1
        dy32 = y3 - y2

        m12 = math.sqrt(math.pow(dx21, 2) + math.pow(dy21, 2))
        m13 = math.sqrt(math.pow(dx31, 2) + math.pow(dy31, 2))
        m23 = math.sqrt(math.pow(dx32, 2) + math.pow(dy32, 2))

        arg_func = (math.pow(m12, 2) + math.pow(m23, 2) - math.pow(m13, 2)) / (2 * m12 * m23)

        # Decimal problem
        # Round to 1 to avoid problems
        if arg_func < -1:
            arg_func = -1
        if arg_func > 1:
            arg_func = 1

        # Law of cosines
        # Check the following link
        # https://www.varsitytutors.com/hotmath/hotmath_help/topics/law-of-cosines
        theta = math.acos(arg_func)

        # No checking pose orientation

        return theta

    @staticmethod
    def check_clockwise(point1, point2, point3):
        # Based on
        # https://www.geeksforgeeks.org/orientation-3-ordered-points/
        # Points must be in ordered

        # print('Checking clockwise order')

        if type(point1) is dict:
            x1 = point1['x']
            y1 = point1['y']
        else:
            x1 = point1[0]
            y1 = point1[1]

        if type(point2) is dict:
            x2 = point2['x']
            y2 = point2['y']
        else:
            x2 = point2[0]
            y2 = point2[1]

        if type(point3) is dict:
            x3 = point3['x']
            y3 = point3['y']
        else:
            x3 = point3[0]
            y3 = point3[1]

        val = (y2 - y1) * (x3 - x2) - (x2 - x1) * (y3 - y2)

        if val == 0:
            # colinear
            return 0
        elif val > 0:
            # clockwise
            return 1
        else:
            # counterclockwise
            return 2

    @staticmethod
    def check_vector_integrity(vector):
        """
        Checking vector integrity
        Vector 1, 2, 5, 8 must exists -> Torse
        Vector 10, 13, must be one -> Legs
        Score is in the 2
        """

        print('Checking vector integrity')
        if vector[1][2] < ClassUtils.MIN_POSE_SCORE:
            return False
        elif vector[2][2] < ClassUtils.MIN_POSE_SCORE:
            return False
        elif vector[5][2] < ClassUtils.MIN_POSE_SCORE:
            return False
        elif vector[8][2] < ClassUtils.MIN_POSE_SCORE:
            return False
        elif vector[10][2] < ClassUtils.MIN_POSE_SCORE and vector[13][2] < ClassUtils.MIN_POSE_SCORE:
            return False
        else:
            return True

    @staticmethod
    def get_rectangle_bounds(person_arr, min_score):
        # Getting rectangle bounds in list
        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0

        first = True
        for item in person_arr:
            if item[2] >= min_score:
                if first:
                    min_x = item[0]
                    min_y = item[1]
                    max_x = item[0]
                    max_y = item[1]
                    first = False
                if item[0] < min_x:
                    min_x = item[0]
                if item[1] < min_y:
                    min_y = item[1]
                if item[0] > max_x:
                    max_x = item[0]
                if item[1] > max_y:
                    max_y = item[1]

        return (int(min_x), int(min_y)), (int(max_x), int(max_y))

    @staticmethod
    def get_rectangle_bounds_upper(person_arr, min_score):
        # Only taking account points 1 to 8
        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0

        first = True
        for index in range(1, 8 + 1):
            item = person_arr[index]
            if item[2] >= min_score:
                if first:
                    min_x = item[0]
                    min_y = item[1]
                    max_x = item[0]
                    max_y = item[1]
                    first = False
                if item[0] < min_x:
                    min_x = item[0]
                if item[1] < min_y:
                    min_y = item[1]
                if item[0] > max_x:
                    max_x = item[0]
                if item[1] > max_y:
                    max_y = item[1]

        return (int(min_x), int(min_y)), (int(max_x), int(max_y))

    @staticmethod
    def check_point_integrity(point, min_pose_score):
        if point[2] < min_pose_score:
            return False
        else:
            return True

    @staticmethod
    def check_point_list(vector_points, min_pose_score):
        result = True

        for point in vector_points:
            if point[2] < min_pose_score:
                result = False
                break

        return result

    @classmethod
    def check_vector_only_pos(cls, vector, min_pose_score):
        # Check if vector is only pos
        # Vector must be checked with check_vector_integrity_pos first
        # If not, an exception will be raised

        if not cls.check_vector_integrity_pos(vector, min_pose_score):
            raise Exception('Vector integrity not valid')

        # Vector integrity part must be used for pose detection
        # Vector integrity pos must be used only for reidentification and position
        valid = cls.check_vector_integrity_part(vector, min_pose_score)

        if valid:
            return False
        else:
            return True

    @classmethod
    def check_vector_integrity_pos(cls, vector, min_pose_score):
        # More relaxed than check_vector_integrity_part
        # Detect position elements
        return cls.check_vector_integrity_part(vector, min_pose_score, only_pos=True)

    @classmethod
    def check_vector_integrity_part(cls, vector, min_pose_score, only_pos=False):
        # Checking vector integrity part
        # One part of the vector must exist

        # Torso must exist
        if vector[1][2] < min_pose_score or vector[8][2] < min_pose_score:
            return False

        # If only pos is activated
        # Arms integrity does not care
        if not only_pos:
            # Check arms
            # One of the segments must be valid
            arms_valid = False
            if vector[2][2] >= min_pose_score and vector[3][2] >= min_pose_score and vector[4][2] >= min_pose_score:
                arms_valid = True
            if vector[5][2] >= min_pose_score and vector[6][2] >= min_pose_score and vector[7][2] >= min_pose_score:
                arms_valid = True

            if not arms_valid:
                return False

        # Check legs
        # One of the segments must be valid
        # There must be one valid leg
        legs_valid = False
        if vector[9][2] >= min_pose_score and vector[10][2] >= min_pose_score and vector[11][2] >= min_pose_score:
            legs_valid = True
        if vector[12][2] >= min_pose_score and vector[13][2] >= min_pose_score and vector[14][2] >= min_pose_score:
            legs_valid = True
        # If not, there must exist points 9, 10, 12, 13 to perform guess points
        # Partial legs
        if vector[9][2] >= min_pose_score and vector[10][2] >= min_pose_score \
                and vector[12][2] >= min_pose_score and vector[13][2] >= min_pose_score:
            legs_valid = True

        if not legs_valid:
            return False

        # Check rectangle bounds
        # If there are not enough rectangle bounds
        # Ignore pose
        pt1, pt2 = cls.get_rectangle_bounds(vector, min_pose_score)
        delta_x = pt2[0] - pt1[0]

        # If delta x is less than threshold, ignore pose!
        # Avoid lum position bug
        if delta_x < 5:
            print('Avoid Frame by low deltaX!')
            return False

        # All check ok - return True
        return True

    @staticmethod
    def get_filename_extension(filename: str):
        extension = os.path.splitext(filename)[1]
        return extension

    @staticmethod
    def get_filename_no_extension(filename: str):
        path_no_ext = os.path.splitext(filename)[0]
        return path_no_ext

    @staticmethod
    def get_ticks(date:datetime):
        t0 = datetime(1, 1, 1)
        seconds = (date - t0).total_seconds()
        ticks = int(seconds * 10 ** 7)
        return ticks

    @staticmethod
    def get_point_intersection(point1_l1, point2_l1, point1_l2, point2_l2):
        # Initializing getting angle lines
        # Getting point intersection
        # Pulled from here: https://stackoverflow.com/questions/4543506/algorithm-for-intersection-of-2-lines

        x1_1 = point1_l1[0]
        x2_1 = point2_l1[0]
        y1_1 = point1_l1[1]
        y2_1 = point2_l1[1]

        x1_2 = point1_l2[0]
        x2_2 = point2_l2[0]
        y1_2 = point1_l2[1]
        y2_2 = point2_l2[1]

        a1 = y2_1 - y1_1
        b1 = x1_1 - x2_1
        c1 = a1 * x1_1 + b1 * y1_1

        a2 = y2_2 - y1_2
        b2 = x1_2 - x2_2
        c2 = a2 * x1_2 + b2 * y1_2

        delta = a1 * b2 - a2 * b1

        if delta == 0:
            # lines are parallel - Raise exception
            raise Exception('Lines are parallel')
        else:
            x = (b2 * c1 - b1 * c2) / delta
            y = (a1 * c2 - a2 * c1) / delta
            intersect_point = [x, y, 1]

            return intersect_point

    @staticmethod
    def get_angle_lines(point1_l1, point2_l1, point1_l2, point2_l2):
        # Translate point and get angle
        # point1_l2 will be translate to point2_l1 location
        # then, the get angle method will be called
        # assume array format

        if type(point1_l1) is dict:
            raise Exception('Format of point1_l1 not supported')
        if type(point2_l1) is dict:
            raise Exception('Format of point2_l1 not supported')
        if type(point1_l2) is dict:
            raise Exception('Format of point1_l2 not supported')
        if type(point2_l2) is dict:
            raise Exception('Format of point2_l2 not supported')

        # Get deltas
        delta_x = point2_l1[0] - point1_l2[0]
        delta_y = point2_l1[1] - point1_l2[1]

        # Transform latest
        new_partial_point = [
            point2_l2[0] + delta_x,
            point2_l2[1] + delta_y,
            1
        ]

        # Call angle point
        angle = ClassUtils.get_angle(point1_l1, point2_l1, new_partial_point)

        return angle

    @staticmethod
    def get_flat_list(n_list):
        # Only works with 2D list
        return [item for sublist in n_list for item in sublist]

    @staticmethod
    def key_to_number(key_cv):
        if key_cv == 48:
            return 0
        elif key_cv == 49:
            return 1
        elif key_cv == 50:
            return 2
        elif key_cv == 51:
            return 3
        elif key_cv == 52:
            return 4
        elif key_cv == 53:
            return 5
        elif key_cv == 54:
            return 6
        elif key_cv == 55:
            return 7
        elif key_cv == 56:
            return 8
        elif key_cv == 57:
            return 9
        else:
            return -1

    @classmethod
    def transform_angle_point(cls, point, center, angle_degrees):
        if type(point) is dict:
            raise Exception('Type point not supported: {0}'.format(type(point)))
        if type(center) is dict:
            raise Exception('Type center not supported: {0}'.format(type(center)))

        # 1 -> Transform in polar coordinates
        r = cls.get_euclidean_distance(point[0], point[1], 0, 0)

        # Avoid zero division and control theta quadrant
        if point[0] != 0:
            theta = math.atan(point[1] / point[0])
        else:
            theta = math.pi / 2

        if point[0] < 0:
            theta += math.pi

        angle = angle_degrees * math.pi / 180
        new_theta = theta + angle

        if new_theta >= 2 * math.pi:
            new_theta -= 2 * math.pi

        # 2 -> Transform in cartesian coordinates
        new_point = list()
        new_point.append(r * math.cos(new_theta))  # x
        new_point.append(r * math.sin(new_theta))  # y
        new_point.append(1)  # confidence

        # 3 -> Perform center adjustment
        new_point[0] += center[0]
        new_point[1] += center[1]

        # 4 -> Perform round to avoid decimal
        new_point[0] = round(new_point[0], 5)
        new_point[1] = round(new_point[1], 5)

        # 4 -> Return point
        return new_point

    @staticmethod
    def get_cam_number_from_path(file_path: str):
        # Getting cam number
        elems = file_path.split('/')
        return elems[-2]

    @staticmethod
    def complete_points(person_vector, min_score):
        # Complete person points
        # Must be set for position purposes

        new_vector = list()
        for point in person_vector:
            new_vector.append([point[0], point[1], point[2]])

        # Complete legs
        # Consider simultaneous cases
        # Vector must be checked first with check_vector_integrity_part
        # Vector 10, 11 are invalid - assume 9 is invalid
        if new_vector[10][2] < min_score and new_vector[11][2] < min_score:
            new_vector[9][0] = new_vector[12][0]
            new_vector[9][1] = new_vector[12][1]
            new_vector[9][2] = new_vector[12][2]
            new_vector[10][0] = new_vector[13][0]
            new_vector[10][1] = new_vector[13][1]
            new_vector[10][2] = new_vector[13][2]
            new_vector[11][0] = new_vector[14][0]
            new_vector[11][1] = new_vector[14][1]
            new_vector[11][2] = new_vector[14][2]
        elif min_score and new_vector[13][2] < min_score and new_vector[14][2] < min_score:
            # Vectors 13 and 14 are invalid - assume 12 is invalid
            new_vector[12][0] = new_vector[9][0]
            new_vector[12][1] = new_vector[9][1]
            new_vector[12][2] = new_vector[9][2]
            new_vector[13][0] = new_vector[10][0]
            new_vector[13][1] = new_vector[10][1]
            new_vector[13][2] = new_vector[10][2]
            new_vector[14][0] = new_vector[11][0]
            new_vector[14][1] = new_vector[11][1]
            new_vector[14][2] = new_vector[11][2]

        # Complete 11 and 14 if invalid
        if new_vector[11][2] < min_score:
            # Project points
            delta_x = new_vector[10][0] - new_vector[9][0]
            delta_y = new_vector[10][1] - new_vector[9][1]
            new_vector[11][0] = new_vector[10][0] + delta_x
            new_vector[11][1] = new_vector[10][1] + delta_y
            new_vector[11][2] = 1
        if new_vector[14][2] < min_score:
            # Project points
            delta_x = new_vector[13][0] - new_vector[12][0]
            delta_y = new_vector[13][1] - new_vector[12][1]
            new_vector[14][0] = new_vector[13][0] + delta_x
            new_vector[14][1] = new_vector[13][1] + delta_y
            new_vector[14][2] = 1

        return new_vector

    @staticmethod
    def _draw_line_pose(image, point0, point1, min_score, color=(255, 255, 0)):
        if point0[2] >= min_score and point1[2] > min_score:
            cv2.line(image, (int(point0[0]), int(point0[1])),
                     (int(point1[0]), int(point1[1])), color, 3)

    @classmethod
    def compute_mean_circular_deg(cls, list_angles_deg):
        list_angle_rad = list()
        for angle in list_angles_deg:
            list_angle_rad.append(angle * math.pi / 180)

        mean_angle_rad = cls.compute_mean_circular(list_angle_rad)
        return mean_angle_rad * 180 / math.pi

    @staticmethod
    def compute_mean(list_numbers):
        result = 0
        for number in list_numbers:
            result += number
        return result / len(list_numbers)

    @staticmethod
    def compute_mean_circular(list_angles_rad: list):
        # Calculate mean using sphere method
        # Inspired in stack overflow response
        # https://stackoverflow.com/questions/491738/how-do-you-calculate-the-average-of-a-set-of-circular-data

        list_points = list()
        x_avg = 0
        y_avg = 0

        for angle in list_angles_rad:
            y_avg += math.sin(angle)
            x_avg += math.cos(angle)

        # Avoid zero division and control theta quadrant
        if x_avg != 0:
            theta = math.atan(y_avg / x_avg)
        else:
            theta = math.pi / 2

        if x_avg < 0:
            theta += math.pi

        if theta < 0:
            theta += 2 * math.pi

        if theta >= 2 * math.pi:
            theta -= 2 * math.pi

        return theta

    @classmethod
    def get_color_diff_bgr(cls, color1_bgr, color2_bgr):
        if isinstance(color1_bgr, np.ndarray):
            color1_item = [int(color1_bgr[0, 0, 0]), int(color1_bgr[0, 0, 1]), int(color1_bgr[0, 0, 2])]
        else:
            color1_item = color1_bgr

        if isinstance(color2_bgr, np.ndarray):
            color2_item = [int(color2_bgr[0, 0, 0]), int(color2_bgr[0, 0, 1]), int(color2_bgr[0, 0, 2])]
        else:
            color2_item = color2_bgr

        return cls.get_color_diff_rgb([color1_item[2], color1_item[1], color1_item[0]],
                                      [color2_item[2], color2_item[1], color2_item[0]])

    @classmethod
    def get_color_diff_rgb(cls, color1_rgb, color2_rgb, eq_lum=False):
        # Inspired in this post
        # http://hanzratech.in/2015/01/16/color-difference-between-2-colors-using-python.html
        # Values must be between 0 and 255

        # Get color diff based on lum equailization
        # Lab color space

        bgr_elem1 = [int(color1_rgb[2]), int(color1_rgb[1]), int(color1_rgb[0])]
        cv_point1 = np.array([[bgr_elem1]], dtype=np.uint8)

        bgr_elem2 = [int(color2_rgb[2]), int(color2_rgb[1]), int(color2_rgb[0])]
        cv_point2 = np.array([[bgr_elem2]], dtype=np.uint8)

        # Convert to lab
        lab1_pt = cv2.cvtColor(cv_point1, cv2.COLOR_BGR2LAB)
        lab2_pt = cv2.cvtColor(cv_point2, cv2.COLOR_BGR2LAB)

        # Equals lab2 lum to lab1
        if eq_lum:
            lab2_pt[0, 0, 0] = lab1_pt[0, 0, 0]

        # Compare and return result
        return cls.get_color_diff_lab(lab1_pt, lab2_pt)

    @classmethod
    def get_color_diff_lum_value(cls, color1_rgb, color2_rgb):
        bgr_elem1 = [int(color1_rgb[2]), int(color1_rgb[1]), int(color1_rgb[0])]
        cv_point1 = np.array([[bgr_elem1]], dtype=np.uint8)

        bgr_elem2 = [int(color2_rgb[2]), int(color2_rgb[1]), int(color2_rgb[0])]
        cv_point2 = np.array([[bgr_elem2]], dtype=np.uint8)

        # Convert to lab
        lab1_pt = cv2.cvtColor(cv_point1, cv2.COLOR_BGR2LAB)
        lab2_pt = cv2.cvtColor(cv_point2, cv2.COLOR_BGR2LAB)

        diff_lum = math.fabs(int(lab2_pt[0, 0, 0]) - int(lab1_pt[0, 0, 0]))

        return diff_lum

    @classmethod
    def get_color_diff_rgb_lum(cls, color1_rgb, color2_rgb):
        # Compare and return result
        return cls.get_color_diff_rgb(color1_rgb, color2_rgb, eq_lum=True)

    @staticmethod
    def get_color_diff_lab(color1_lab, color2_lab):

        if isinstance(color1_lab, np.ndarray):
            color1_item = [int(color1_lab[0, 0, 0]), int(color1_lab[0, 0, 1]), int(color1_lab[0, 0, 2])]
        else:
            color1_item = color1_lab

        if isinstance(color2_lab, np.ndarray):
            color2_item = [int(color2_lab[0, 0, 0]), int(color2_lab[0, 0, 1]), int(color2_lab[0, 0, 2])]
        else:
            color2_item = color2_lab

        color1 = LabColor(color1_item[0] * 100 / 255, color1_item[1] - 128, color1_item[2] - 128)
        color2 = LabColor(color2_item[0] * 100 / 255, color2_item[1] - 128, color2_item[2] - 128)

        # Find the color difference
        delta_e = delta_e_cie2000(color1, color2)
        return delta_e

    @staticmethod
    def generate_uuid():
        guid = str(uuid.uuid4())
        return guid

    @staticmethod
    def check_valid_vector_points(person_vector, min_score):
        vector_count = 0
        for point in person_vector:
            if point[2] >= min_score:
                vector_count += 1

        return vector_count

    @staticmethod
    def cv_key_to_number(key):
        if key == 48:
            return 0
        elif key == 49:
            return 1
        elif key == 50:
            return 2
        elif key == 51:
            return 3
        elif key == 52:
            return 4
        elif key == 53:
            return 5
        elif key == 54:
            return 6
        elif key == 55:
            return 7
        elif key == 56:
            return 8
        elif key == 57:
            return 9
        else:
            return -1

    @staticmethod
    def eq_lum_rgb_colors(color_rgb, color_dst_rgb):
        # Insert arr
        bgr_elem1 = [int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0])]
        cv_point1 = np.array([[bgr_elem1]], dtype=np.uint8)

        bgr_elem2 = [int(color_dst_rgb[2]), int(color_dst_rgb[1]), int(color_dst_rgb[0])]
        cv_point2 = np.array([[bgr_elem2]], dtype=np.uint8)

        # Convert to lab
        lab1_pt = cv2.cvtColor(cv_point1, cv2.COLOR_BGR2LAB)
        lab2_pt = cv2.cvtColor(cv_point2, cv2.COLOR_BGR2LAB)

        # Equals lab2 lum to lab1
        lab2_pt[0, 0, 0] = lab1_pt[0, 0, 0]

        # Transform again in bgr
        eq_bgr_color = cv2.cvtColor(lab2_pt, cv2.COLOR_LAB2BGR)

        # Returns arr
        return [int(eq_bgr_color[0, 0, 2]), int(eq_bgr_color[0, 0, 1]), int(eq_bgr_color[0, 0, 0])]

    @classmethod
    def compare_colors(cls, upper1_rgb, upper2_rgb, lower1_rgb, lower2_rgb):
        bgr_upper1 = [int(upper1_rgb[2]), int(upper1_rgb[1]), int(upper1_rgb[0])]
        cv_upper1 = np.array([[bgr_upper1]], dtype=np.uint8)

        bgr_upper2 = [int(upper2_rgb[2]), int(upper2_rgb[1]), int(upper2_rgb[0])]
        cv_upper2 = np.array([[bgr_upper2]], dtype=np.uint8)

        bgr_lower1 = [int(lower1_rgb[2]), int(lower1_rgb[1]), int(lower1_rgb[0])]
        cv_lower1 = np.array([[bgr_lower1]], dtype=np.uint8)

        bgr_lower2 = [int(lower2_rgb[2]), int(lower2_rgb[1]), int(lower2_rgb[0])]
        cv_lower2 = np.array([[bgr_lower2]], dtype=np.uint8)

        # Convert to lab
        lab_upper1 = cv2.cvtColor(cv_upper1, cv2.COLOR_BGR2LAB)
        lab_upper2 = cv2.cvtColor(cv_upper2, cv2.COLOR_BGR2LAB)
        lab_lower1 = cv2.cvtColor(cv_lower1, cv2.COLOR_BGR2LAB)
        lab_lower2 = cv2.cvtColor(cv_lower2, cv2.COLOR_BGR2LAB)

        # Calculate delta
        delta_lower = int(lab_lower2[0, 0, 0]) - int(lab_lower1[0, 0, 0])

        # Add candidates
        list_upper_cn = list()
        list_lower_cn = list()

        list_upper_cn.append((lab_upper1, lab_upper2, 0))
        list_lower_cn.append((lab_lower1, lab_lower2, 0))

        # Add elements into list - Lower_delta
        # Only if it is less than 100
        if delta_lower < 100:
            list_upper_cn.append((cls.add_delta_lab(lab_upper1, delta_lower), lab_upper2, delta_lower))
            list_lower_cn.append((cls.add_delta_lab(lab_lower1, delta_lower), lab_lower2, delta_lower))

        min_score = -1
        min_diff_upper = 0
        min_diff_lower = 0
        min_delta = 0

        for i in range(len(list_upper_cn)):
            upper1, upper2, delta = list_upper_cn[i]
            lower1, lower2, _ = list_lower_cn[i]

            diff_upper = cls.get_color_diff_lab(upper1, upper2)
            diff_lower = cls.get_color_diff_lab(lower1, lower2)

            score = diff_upper
            if min_score == -1 or score < min_score:
                min_score = score
                min_diff_upper = diff_upper
                min_diff_lower = diff_lower
                min_delta = delta

        return min_diff_upper, min_diff_lower, min_delta

    @staticmethod
    def add_delta_lab(np_lab, delta):
        new_color = np.copy(np_lab)

        new_lum = int(new_color[0, 0, 0] + delta)
        if new_lum > 255:
            new_lum = 255
        elif new_lum < 0:
            new_lum = 0

        new_color[0, 0, 0] = new_lum

        return new_color

    @classmethod
    def write_in_file(cls, file, ticks, image, json_dict=None):
        bytes_to_write = cls.get_bytes_file(ticks, image, json_dict)
        file.write(bytes_to_write)

    @staticmethod
    def get_bytes_file(ticks, image, json_dict=None):
        # If json_dict exist
        # File is mjpegx

        len_json = 0
        len_json_bin = bytes()
        json_bytes = bytes()  # Len 0

        # Support mjpeg and mjpegx conversion
        if json_dict is not None:
            json_string = json.dumps(json_dict)
            json_bytes = bytes(json_string, encoding='utf-8')
            len_json = len(json_bytes)
            len_json_bin = len_json.to_bytes(length=4, byteorder='little')

        len_total = len(image)
        if json_dict is not None:
            len_total += len_json + 4  # 4 -> 4 is the len of json data

        len_total_bin = len_total.to_bytes(length=4, byteorder='little')

        ticks_bin = ticks.to_bytes(length=8, byteorder='little')

        result = len_total_bin + ticks_bin + image

        # Write mjpegx part
        if json_dict is not None:
            result += json_bytes + len_json_bin

        return result

    @staticmethod
    def get_date_file(date):
        minutes = int(date.minute / 15) * 15
        seconds = 0
        microseconds = 0
        date_file = date.replace(minute=minutes, second=seconds, microsecond=microseconds)
        return date_file

    @classmethod
    def load_path_by_date(cls, date: datetime, cam_number: str, extension: str):
        logger.debug('Generating path by date')
        file_name = date.strftime('%H-%M-%S') + extension
        file_path = path.join(cls.video_base_path, date.strftime('%Y-%m-%d'), str(cam_number), file_name)
        logger.debug(file_path)
        return file_path

    @classmethod
    def equalize_hist(cls, image: np.ndarray):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Equalize hist from YUV channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

    @classmethod
    def blur(cls, image: np.ndarray, size: int):
        if size % 2 == 0:
            raise Exception('size must be an odd number!')

        # Blurring image using kernel
        kernel = np.ones((size, size), np.float32) / (size * size)

        dst = cv2.filter2D(image, -1, kernel)
        return dst

    @classmethod
    def change_ext_training(cls, file_name: str, new_ext_train: str):
        list_elem = file_name.split('_')
        ext = cls.get_filename_extension(file_name)

        new_file_name = ''
        for i in range(len(list_elem) - 1):
            if i == 0:
                new_file_name = list_elem[i]
            else:
                new_file_name += '_'
                new_file_name += list_elem[i]

        new_file_name += '_'
        new_file_name += new_ext_train
        new_file_name += ext

        return new_file_name
