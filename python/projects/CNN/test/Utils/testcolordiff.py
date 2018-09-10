from classutils import ClassUtils
import cv2
import numpy as np


def main():
    print('Initializing main function')

    color1 = [158, 145, 179]
    color2 = [70, 73, 93]

    delta_e = ClassUtils.get_color_diff_rgb(color1, color2)
    print('Color diff: {0}'.format(delta_e))

    # Showing colors
    # In 100 x 100 window
    # Trying to compare colors visually
    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)

    img_size = 100
    image1 = np.zeros((img_size, img_size, 3), np.uint8)
    image2 = np.zeros((img_size, img_size, 3), np.uint8)

    # BGR format -> Generating
    image1[:, :] = (color1[2], color1[1], color1[0])
    image2[:, :] = (color2[2], color2[1], color2[0])

    # Showing image stacked
    img = np.hstack((image1, image2))
    cv2.imshow('main_window', img)

    print('Press a key to continue')
    cv2.waitKey(0)
    print('Done!')


if __name__ == '__main__':
    main()
