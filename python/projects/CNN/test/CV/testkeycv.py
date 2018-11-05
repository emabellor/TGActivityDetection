import cv2


def main():
    print('Initializing main function')

    image_file = '/home/mauricio/Pictures/2.jpg'
    image = cv2.imread(image_file)

    cv2.namedWindow('mainWindow', cv2.WND_PROP_AUTOSIZE)
    cv2.imshow('mainWindow', image)

    print('Press a key')
    while True:
        key = cv2.waitKey(100)

        if key != -1:
            print('Key Pressed: {0}'.format(key))

            # Escape
            if key == 27:
                break

    cv2.destroyAllWindows()
    print('Done!')


if __name__ == '__main__':
    main()

