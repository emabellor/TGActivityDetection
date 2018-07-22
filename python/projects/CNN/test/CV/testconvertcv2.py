import cv2
import numpy as np


def main():
    print('Init test convert cv2 - Loading file')
    image = cv2.imread('/home/mauricio/Pictures/429.jpg', cv2.IMREAD_GRAYSCALE)  # type: np.ndarray

    print('Printing array')
    print(image)

    print('Image read. Printing dimensions')
    print(image.ndim)

    print('Printing type')
    print(image.dtype)

    print('Converting type to float')
    image = image.astype(float)

    print(image.dtype)

    print('Printing array')
    print(image)

    print('Done!')


if __name__ == '__main__':
    main()
