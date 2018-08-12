from classdescriptors import ClassDescriptors


def main():
    print('Initializing main function')

    # Loading base points
    point_1 = [2, 2, 1]
    point_2 = [0, 3, 1]
    point_3 = [2, 0, 1]
    point_4 = [1, 0, 1]

    # Ignore protected warnings
    point_g1 = ClassDescriptors._guess_leg_point(point_1, point_2)
    point_g2 = ClassDescriptors._guess_leg_point(point_3, point_4)

    print('Point g1: {0} - Point g2: {1}'.format(point_g1, point_g2))
    print('Done!')


if __name__ == '__main__':
    main()
