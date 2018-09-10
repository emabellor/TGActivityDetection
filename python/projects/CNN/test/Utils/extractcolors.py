import cv2
import numpy as np


def main():
    print('Initializing main function')

    # Creating virtual image
    # Images in opencv are in bgr format
    img_size = 100
    image = np.zeros((img_size, img_size, 3), np.uint8)

    color = [255, 0, 0]
    image[:, :] = (color[2], color[1], color[0])

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    cv2.imshow('main_window', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Done!')


if __name__ == '__main__':
    main()
