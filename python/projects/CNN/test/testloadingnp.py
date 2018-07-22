# The array could be loaded successfully
import numpy as np


def main():
    print('Test loading np')
    path_numpy = '/home/mauricio/Datasets/example.npy'
    image_res = np.load(path_numpy)

    print('Trying to get shape')
    print(image_res.shape)

    try:
        print('Trying to load wrong array')
        path_wrong = '/home/mauricio2/Datasets/example.npy'
        image_res = np.load(path_wrong)
        print(image_res.shape)
    except FileNotFoundError as ex:
        print('Exception thrown: ' + ex.strerror)

    print('Done!')


if __name__ == '__main__':
    main()