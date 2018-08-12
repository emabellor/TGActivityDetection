from classutils import ClassUtils
import math


def main():
    print('Initializing main function')

    print('Loading angle points')
    point_1 = [2, 2, 1]
    point_2 = [0, 0, 1]
    point_3 = [2, 0, 1]
    point_4 = [1, 0, 1]

    angle1 = ClassUtils.get_angle(point_1, point_2, point_3) * 180 / math.pi
    angle2 = ClassUtils.get_angle_lines(point_1, point_2, point_2, point_3) * 180 / math.pi
    angle3 = ClassUtils.get_angle_lines(point_1, point_2, point_3, point_4) * 180 / math.pi
    angle4 = ClassUtils.get_angle_lines(point_1, point_2, point_4, point_3) * 180 / math.pi

    print('Angle 1: {0} - Angle 2: {1} - Angle 3: {2} - Angle 4: {3}'.format(angle1, angle2, angle3, angle4))

    print('Done!')


if __name__ == '__main__':
    main()
