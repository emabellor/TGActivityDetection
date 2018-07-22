import numpy as np


def main():
    print('Testing numpy operations')

    array = np.array([[1, 2, 3], [4, 5, 6]])
    print(array)

    print('Dividing by value')
    array = array / 222

    print(array)

    print('Getting elements in list')
    print(array.shape[0])

    print('Done!')


if __name__ == '__main__':
    main()
