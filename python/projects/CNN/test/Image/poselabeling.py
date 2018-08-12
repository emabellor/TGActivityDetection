from tkinter import Tk
from tkinter import filedialog
import cv2
import os
import shutil


def main():
    print('Initializing main function')

    # Withdrawing tkinter
    Tk().withdraw()

    # Loading folder images
    folder_images = '/home/mauricio/PosesProcessed/folder_images'
    folder_images_draw = '/home/mauricio/PosesProcessed/folder_images_draw'
    pose_base_folder = '/home/mauricio/Pictures/Poses'

    # Reading all elements in list
    list_files = sorted(os.listdir(folder_images_draw))

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    index = 0

    list_poses = [
        'walking_left_0',
        'walking_left_1',
        'walking_left_2',
        'walking_right_0',
        'walking_right_1',
        'walking_right_2',
    ]

    while True:
        file = list_files[index]
        full_path = os.path.join(folder_images_draw, file)

        image = cv2.imread(full_path)
        cv2.imshow('main_window', image)

        key = cv2.waitKey(0)
        print('Key pressed: {0}'.format(key))
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
        elif key == 115:
            init_dir = '/home/mauricio/Pictures/Poses'
            options = {'initialdir': init_dir}
            dir_name = filedialog.askdirectory(**options)

            if not dir_name:
                print('Directory not selected')
            else:
                new_full_path = os.path.join(dir_name, os.path.basename(full_path))
                shutil.copyfile(os.path.join(folder_images, file), new_full_path)

                print('File copied from {0} to {1}'.format(os.path.join(folder_images, file), new_full_path))
                index += 1
                if index == len(list_files):
                    index = len(list_files) - 1

    cv2.destroyAllWindows()
    print('Done!')


if __name__ == '__main__':
    main()
