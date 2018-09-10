from classutils import ClassUtils
import cv2
import numpy as np
from classcolorcomparison import ClassColorComparison


def main():
    print('Initializing main function')

    color1 = [199, 164, 112]
    color2 = [128, 112, 132]

    delta_e = ClassUtils.get_color_diff_rgb(color1, color2)
    print('Color diff: {0}'.format(delta_e))

    # Showing colors
    # In 100 x 100 window
    # Trying to compare colors visually
    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)

    img_size = 100
    image1_1 = np.zeros((img_size, img_size, 3), np.uint8)
    image2_1 = np.zeros((img_size, img_size, 3), np.uint8)

    # BGR format -> Generating
    image1_1[:, :] = (color1[2], color1[1], color1[0])
    image2_1[:, :] = (color2[2], color2[1], color2[0])

    # Showing image stacked
    img = np.hstack((image1_1, image2_1))

    result1 = ClassColorComparison.classify_color(color1)
    result2 = ClassColorComparison.classify_color(color2)

    image1_2 = np.zeros((img_size, img_size, 3), np.uint8)
    image2_2 = np.zeros((img_size, img_size, 3), np.uint8)

    image1_2[:, :] = (result1[1][2], result1[1][1], result1[1][0])
    image2_2[:, :] = (result2[1][2], result2[1][1], result2[1][0])

    img_aux = np.hstack((image1_2, image2_2))
    img_result = np.vstack((img, img_aux))

    cv2.imshow('main_window', img_result)

    print('Press a key to continue')
    cv2.waitKey(0)
    print('Done!')


if __name__ == '__main__':
    main()
