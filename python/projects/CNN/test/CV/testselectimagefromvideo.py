"""
Script to select images from video
"""


from tkinter import Tk
from tkinter.filedialog import askopenfilename
from classmjpegreader import ClassMjpegReader
import cv2
import numpy as np
import os


def main():
    print('Selecting file')

    text = input('Insert camera number: ')

    print('Select video from folder')
    Tk().withdraw()

    init_dir = '/home/mauricio/Videos/Oviedo/2018-10-30/' + text
    if not os.path.isdir(init_dir):
        init_dir = '/home/mauricio/Videos/Oviedo/'

    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if not filename:
        print('File not selected')
    else:
        print('Loading mjpeg elems')

        frame_list = ClassMjpegReader.process_video(filename)
        cv2.namedWindow('image')
        for frame in frame_list:
            image = frame[0]
            ticks = frame[1]

            print('Showing image -> press Enter to accept, q to take next, esc to quit')
            image_cv = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow('image', image_cv)
            take_next = True

            while True:
                k = cv2.waitKey(100)

                if k == 13:  # enter
                    base_dir = '/home/mauricio/Oviedo/CameraCalibration/' + text + '/image_' + text + '.jpg'

                    folder_dir = os.path.dirname(base_dir)

                    if not os.path.exists(folder_dir):
                        os.makedirs(folder_dir)

                    print('Saving image ' + base_dir)

                    with open(base_dir, 'wb') as file:
                        file.write(image)

                    take_next = False
                    break

                if k == 113:  # q
                    print('Taking next')
                    break

                if k == 27:
                    print('Finishing')
                    take_next = False
                    break

            if not take_next:
                break

        print('Done!')


if __name__ == '__main__':
    main()
