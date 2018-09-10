from classutils import ClassUtils
import math


def main():
    print('Initializing main function')

    command = input('Select 1 for test angles. 2 to test clockwise. Otherwise to exit: ')

    if command == '1':
        test_angles()
    elif command == '2':
        test_clockwise_detector()
    else:
        print('Exit!')


def test_angles():
    print('Creating angle test')

    point1 = {
        'x': -4,
        'y': 4
    }

    point2 = {
        'x': 0,
        'y': 0
    }

    point3 = {
        'x': 2,
        'y': 0
    }


    print('Getting angle')
    theta = ClassUtils.get_angle(point1, point2, point3)

    print('Angle: {0}'.format(theta))

    degrees = theta * 180 / math.pi
    print('Angle degrees: {0}'.format(degrees))

    print('Done!')


def test_clockwise_detector():
    print('Testing clockwise detector')

    point1 = {
        'x': 2,
        'y': 0
    }

    point2 = {
        'x': 0,
        'y': 0
    }

    point3 = {
        'x': -4,
        'y': 4
    }

    clockwise = ClassUtils.check_clockwise(point1, point2, point3)
    print('Clockwise result: {0}'.format(clockwise))

    print('Done!')


if __name__ == '__main__':
    main()
