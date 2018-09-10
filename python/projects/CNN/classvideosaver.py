"""
Based on ClassMjpegConverter and ClassMjpegDate classes
"""

from classutils import ClassUtils
from classmjpegconverter import ClassMjpegConverter
from classmjpegdate import ClassMjpegDate
from datetime import datetime
import numpy as np
import os

buffer_size = 1 * 1024 * 1024


class ClassVideoSaver:
    def __init__(self, _cam_number):
        self.buffer = bytes()
        self.current_date_file = datetime.now()
        self.cam_number = str(_cam_number)

    def add_frame(self, image, date_image, json_dict=None):

        if isinstance(image, np.ndarray):
            image_bin = image.tobytes()
        else:
            image_bin = image

        date_file = ClassMjpegDate.get_date_file(date_image)

        if date_file != self.current_date_file:
            self.save_data()
            self.current_date_file = date_file

        ticks = ClassUtils.datetime_to_ticks(date_image)
        file_bytes = ClassMjpegConverter.get_bytes_file(ticks, image_bin, json_dict)

        if len(self.buffer) + len(file_bytes) >= buffer_size:
            self.save_data()

        self.buffer += file_bytes

    def save_data(self):
        if len(self.buffer) > 0:
            ext = '.mjpeg'
            path_file = ClassMjpegDate.load_path_by_date(self.current_date_file, self.cam_number, ext)

            path_folder = os.path.dirname(path_file)

            # Creating path folder if does not exist
            if not os.path.exists(path_folder):
                os.makedirs(path_folder)

            with open(path_file, 'ab') as f:
                f.write(self.buffer)

            # Reset buffer
            self.buffer = bytes()




