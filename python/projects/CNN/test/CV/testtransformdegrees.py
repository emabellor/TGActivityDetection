"""
File to test transform_angle_point from ClassUtils
"""
from classutils import ClassUtils


def main():
    print('Initializing main function')

    # Loading instances
    center = [200, 200]
    point = [1, 2]

    angle1 = 0
    angle2 = 180
    angle3 = 270

    new_point = ClassUtils.transform_angle_point(point, center, angle1)
    print('Point1: {0}'.format(new_point))

    new_point = ClassUtils.transform_angle_point(point, center, angle2)
    print('Point2: {0}'.format(new_point))

    new_point = ClassUtils.transform_angle_point(point, center, angle3)
    print('Point3: {0}'.format(new_point))

    print('Done!')


if __name__ == '__main__':
    main()
