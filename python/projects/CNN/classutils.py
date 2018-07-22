import numpy as np
import os
import json
from datetime import datetime
from datetime import timedelta
import math


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

    @staticmethod
    def load_homo_mat(camera_number: str):
        base_dir = '/home/mauricio/Oviedo/CameraCalibration/' + camera_number + '/calibration.json'

        if not os.path.exists(base_dir):
            raise Exception(base_dir + ' does not exist in system')
        else:
            with open(base_dir, 'r') as content_file:
                config = content_file.read()

            dict_config = json.loads(config)
            homo_mat = np.asarray(dict_config['homographyMat'], dtype='float')

            return homo_mat

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