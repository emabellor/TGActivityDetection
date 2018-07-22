from classmjpegreader import ClassMjpegReader
from classopenpose import ClassOpenPose
import cv2
import numpy as np


def main():
    print('Initializing main function')
    file_path = '/home/mauricio/Videos/mjpeg/11-00-00.mjpeg'

    print('Initializing instance')
    open_pose = ClassOpenPose()

    images = ClassMjpegReader.process_video(file_path)
    print('Size list')
    print(len(images))
    print('Done')

    print('Reading list using opencv')
    cv2.namedWindow('test')

    for elem in images:
        image = elem[0]
        ticks = elem[1]

        image_np = np.frombuffer(image, dtype="int32")
        print(ticks)
        print(len(image))
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_ANYCOLOR)
        image_recognize = open_pose.recognize_image_draw(image_cv)

        cv2.imshow('test', image_recognize)
        cv2.waitKey(10)

    cv2.destroyAllWindows()
    print('Done!')


if __name__ == '__main__':
    main()
