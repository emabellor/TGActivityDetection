"""
Written by
Eder Mauricio Abello
"""
import numpy as np
import cv2


class TFHandler:

    def __init__(self):
        pass

    @staticmethod
    def get_numpy_array(image_list):
        """Get image list. All images must be in jpeg format"""
        temp = []

        for img_name in image_list:
            img = cv2.imread(img_name)

            if img is None:
                raise ValueError(img_name, ' is not an image')

            temp.append(img)

        all_images = np.asarray(temp, dtype=np.float32)

        return all_images
