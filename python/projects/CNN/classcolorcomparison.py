# Inspired in
# https://www.npmjs.com/package/color-classifier
import cv2
from classutils import ClassUtils


class ClassColorComparison:
    W3C = [
        [0x00, 0x00, 0x00],
        [0x80, 0x80, 0x80],
        [0xC0, 0xC0, 0xC0],
        [0xFF, 0xFF, 0xFF],
        [0x80, 0x00, 0x00],
        [0xFF, 0x00, 0x00],
        [0x00, 0x80, 0x00],
        [0x00, 0xFF, 0x00],
        [0x80, 0x80, 0x00],
        [0xFF, 0xFF, 0x00],
        [0x00, 0x80, 0x80],
        [0x00, 0xFF, 0xFF],
        [0x00, 0x00, 0x80],
        [0x00, 0x00, 0xFF],
        [0x80, 0x00, 0x80],
        [0xFF, 0x00, 0xFF]
    ]

    @classmethod
    def classify_color(cls, color_rgb):
        min_score = -1
        min_index = 0

        for index, color_palette in enumerate(cls.W3C):
            diff = ClassUtils.get_color_diff_rgb(color_palette, color_rgb)

            if min_score == -1 or diff < min_score:
                min_score = diff
                min_index = index

        return min_index, cls.W3C[min_index]

