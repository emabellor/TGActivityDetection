from classopenpose import ClassOpenPose
from classdescriptors import ClassDescriptors
import cv2

min_score = 0.05


def main():
    print('Initializing main function')

    # Initialing instances
    instance_pose = ClassOpenPose()

    filename = '/home/mauricio/Pictures/walk.jpg'
    image = cv2.imread(filename)

    arr = instance_pose.recognize_image(image)

    if len(arr) != 1:
        raise Exception('There is more than one person in the image: {0}'.format(len(arr)))

    per_arr = arr[0]
    print(per_arr)

    ClassDescriptors.draw_pose(image, per_arr, min_score)
    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)

    cv2.imshow('main_window', image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    print('Done!')


if __name__ == '__main__':
    main()
