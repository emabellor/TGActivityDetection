"""
Testing opencv homography
"""

import cv2
import numpy as np


def main():
    print('Initializing main function')
    image_points = np.array([[110, 100],
                             [190, 100],
                             [100, 200],
                             [200, 200]], dtype='float32')

    object_points = np.array([[500, 500],
                             [510, 500],
                             [500, 510],
                             [510, 510]], dtype='float32')

    print('Getting homography')
    matrix, mask = cv2.findHomography(image_points, object_points)

    # Transform object points
    print(object_points.shape[0])
    print(object_points.shape[1]+1)
    new_image_points = np.ones((object_points.shape[0], object_points.shape[1]+1), dtype='float32')
    new_image_points[:, :-1] = image_points

    for i in range(4):
        projected_point = np.matmul(matrix, new_image_points[i])
        projected_point[0] = projected_point[0] / projected_point[2]
        projected_point[1] = projected_point[1] / projected_point[2]
        projected_point[2] = 1
        print(projected_point)

    print('Done!')


if __name__ == '__main__':
    main()
