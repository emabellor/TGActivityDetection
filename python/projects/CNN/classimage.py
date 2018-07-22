import cv2
import numpy as np


class ClassImage:

    @staticmethod
    def load(path: str) -> np.ndarray:
        # Loading image normal
        return cv2.imread(path)

    @staticmethod
    def load_gray(path: str) -> np.ndarray:
        # Loading image gray scale
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def resize(image: np.ndarray, img_size) -> np.ndarray:
        return cv2.resize(image, img_size)

