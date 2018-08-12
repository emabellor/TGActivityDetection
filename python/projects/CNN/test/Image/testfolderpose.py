from tkinter import Tk
from tkinter import filedialog
import os
import cv2
from classopenpose import ClassOpenPose


def main():
    print('Initializing main function')

    # Initializing instances
    instance_pose = ClassOpenPose()

    # Withdrawing Tk window
    Tk().withdraw()

    # Loading folder from element in list
    init_dir = '/home/mauricio/Pictures/Poses'
    options = {'initialdir': init_dir}
    dir_name = filedialog.askdirectory(**options)

    if not dir_name:
        print('Directory not selected')
    else:
        print('Selected dir: {0}'.format(dir_name))

        list_files = os.listdir(dir_name)
        cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)

        index = 0
        print('Press arrows to move, ESC to exit')
        while True:
            file = list_files[index]
            full_path = os.path.join(dir_name, file)

            print('Loading image {0}'.format(full_path))
            image = cv2.imread(full_path)

            arr, pro_img = instance_pose.recognize_image_tuple(image)
            cv2.imshow('main_window', pro_img)

            key = cv2.waitKey(0)

            if key == 52:
                # Left arrow
                index -= 1
                if index < 0:
                    index = 0
            elif key == 54:
                # Right arrow
                index += 1
                if index == len(list_files):
                    index = len(list_files) - 1
            elif key == 27:
                # Esc
                break

        print('Done!')


if __name__ == '__main__':
    main()
