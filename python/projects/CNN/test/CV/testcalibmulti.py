import cv2
import numpy as np
from classutils import ClassUtils


def main():
    print('Initializing main function')

    # Defining points
    image_points = list()
    object_points = list()

    image_points.append([0, 0])
    image_points.append([100, 0])
    image_points.append([10, 100])
    image_points.append([90, 100])

    object_points.append([0, 0])
    object_points.append([200, 0])
    object_points.append([0, 200])
    object_points.append([200, 200])

    # Find homography using opencv functions
    matrix, mask = cv2.findHomography(np.array(image_points), np.array(object_points))

    for point in image_points:
        result = ClassUtils.project_points(matrix, np.array(point, dtype=np.int))
        print('Projected points: {0}'.format(result))

    print('Done!')


if __name__ == '__main__':
    main()
