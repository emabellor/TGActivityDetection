import numpy as np
import os
import json
from datetime import datetime
from datetime import timedelta
import math
from datetime import datetime


class ClassUtils:
    # Class Variables
    MIN_POSE_SCORE = 0.1

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
        return os.path.exists(base_dir)

    @staticmethod
    def ticks_to_datetime(ticks):
        dt = datetime(1, 1, 1) + timedelta(microseconds=ticks / 10)
        return dt

    @staticmethod
    def get_euclidean_distance(x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

    @staticmethod
    def write_bin_to_file(path_file: str, bin_array):
        with open(path_file, 'wb') as newFile:
            newFile.write(bin_array)

    @staticmethod
    def get_euclidean_point(point1, point2):
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

        return ClassUtils.get_euclidean_distance(x1, y1, x2, y2)

    @staticmethod
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
        Score is in the position 2
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
    def check_vector_integrity_low(vector, min_pose_score):
        """
        Checking vector integrity
        Vector 1, 8 must exists -> Torse
        Vector 2, 5 must be one -> Shoulders
        Vector 10, 13, must be one -> Legs
        Score is in the position 2
        """

        if vector[1][2] < min_pose_score:
            return False
        elif vector[2][2] < min_pose_score and vector[5][2] < min_pose_score:
            return False
        elif vector[8][2] < min_pose_score:
            return False
        elif vector[10][2] < min_pose_score and vector[13][2] < min_pose_score:
            return False
        else:
            return True

    @staticmethod
    def check_point_integrity(point, min_pose_score):
        if point[2] < min_pose_score:
            return False
        else:
            return True

    @staticmethod
    def check_vector_integrity_full(vector, min_pose_score):
        # Checking vector integrity from all elements
        # Elements from 1 to 14
        # Total elements: 14
        result = True

        for i in range(14):
            if vector[i + 1][2] < min_pose_score:
                result = False
                break

        return result

    @staticmethod
    def check_point_list(vector_points, min_pose_score):
        result = True

        for point in vector_points:
            if point[2] < min_pose_score:
                result = False
                break

        return result

    @staticmethod
    def check_vector_integrity_part(vector, min_pose_score):
        # Checking vector integrity part
        # One part of the vector must exist

        # Torso must exist
        if vector[1][2] < min_pose_score or vector[8][2] < min_pose_score:
            return False

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
