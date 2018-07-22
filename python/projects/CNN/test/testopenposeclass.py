from classopenpose import ClassOpenPose
import cv2


def main():
    print('Initializing main function')
    print('Initializing instance')

    instance = ClassOpenPose()
    print('Reading image')

    image = cv2.imread('/home/mauricio/Pictures/person.jpg')
    image2 = cv2.imread('/home/mauricio/Pictures/430.jpg')

    print('Recognizing image 1')
    arr = instance.recognize_image(image)
    print(arr)

    print('Recognizing image 2')
    arr = instance.recognize_image(image2)
    print(arr)

    print('Done generating elements!')


if __name__ == '__main__':
    main()
