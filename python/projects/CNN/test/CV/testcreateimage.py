import cv2
import numpy as np


def main():
    print('Generating opencv image')

    rows = 500
    cols = 1000

    print('{0} x {1} image'.format(rows, cols))

    blank_image = np.zeros((rows, cols, 3), np.uint8)

    # Test BGR components
    # As opencv works
    for i in range(rows):
        for j in range(cols):
            # Set red image
            blank_image[i, j] = [0, 0, 255]

    cv2.namedWindow('mainWindow', cv2.WND_PROP_AUTOSIZE)
    cv2.imshow('mainWindow', blank_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Done!')


if __name__ == '__main__':
    main()
