from classutils import ClassUtils
import math


def main():
    print('Initializing main function')
    list1 = [359, 0, 1]

    mean = ClassUtils.compute_mean_circular_deg(list1)
    print('Mean list 1: {0}'.format(mean))

    list1 = [180, 90, 1]
    mean = ClassUtils.compute_mean_circular_deg(list1)
    print('Mean list 1: {0}'.format(mean))

    list_rad = [math.pi, 3 * math.pi / 2, 0]
    mean = ClassUtils.compute_mean_circular(list_rad)
    print('Mean list rad: {0}'.format(mean))


if __name__ == '__main__':
    main()
