"""Written by
Mauricio Abello"""

import numpy as np
import cv2


def main():
    """Main function"""
    print('Initializing...')
    img = cv2.imread('apple.jpg', 0)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
