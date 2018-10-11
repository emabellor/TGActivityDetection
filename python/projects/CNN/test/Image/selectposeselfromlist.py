from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json
import cv2
from classutils import ClassUtils
from classdescriptors import ClassDescriptors
import numpy as np
from sys import platform
import os


def main():
    print('Initializing main function')
    Tk().withdraw()

    # Selection
    init_dir = '/home/mauricio/Pictures/CNN'
    if platform == 'win32':
        init_dir = 'C:\\SharedFTP'

    # Base folder
    base_folder = '/home/mauricio/Pictures/BasePoses'
    if platform == 'win32':
        base_folder = 'C:\\SharedFTP\\BaseFolder'

    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if not filename:
        raise Exception('Filename not selected!')

    print('Selected filename: {0}'.format(filename))

    with open(filename, 'r') as f:
        json_str = f.read()

    list_poses = json.loads(json_str)['listPoses']

    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)
    index = 0
    while True:
        elem = list_poses[index]

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image.fill(255)

        vectors = elem['vectors']
        pose_guid = elem['poseGuid']
        min_score = 0.05
        ClassDescriptors.draw_pose(image, vectors, min_score)

        cv2.imshow('main_window', image)

        print('Press esc to quit. S to save image. Left arrow to go left. Right arrow to go right.')
        key = cv2.waitKey(0)

        if key != -1:
            print('Key pressed: {0}'.format(key))

            if key == 27:
                break
            elif key == 52:
                index -= 1
                if index < 0:
                    index = 0
            elif key == 54:
                index += 1
                if index > len(list_poses) - 1:
                    index = len(list_poses) - 1
            elif key == 115:
                # Saving image
                image_path = os.path.join(base_folder, pose_guid + '.jpg')
                data_path = os.path.join(base_folder, pose_guid + '.json')

                # Checking if directory exists
                if not os.path.exists(base_folder):
                    os.makedirs(base_folder)

                cv2.imwrite(image_path, image)
                with open(data_path, 'w') as f:
                    f.write(json.dumps(elem))

                print('Image saved: {0}'.format(data_path))
                index += 1
            elif key == 97:
                # Loading image again
                options = {'initialdir': init_dir}
                filename = askopenfilename(**options)

                if not filename:
                    raise Exception('Filename not selected!')

                print('Selected filename: {0}'.format(filename))

                with open(filename, 'r') as f:
                    json_str = f.read()

                list_poses = json.loads(json_str)['listPoses']
                index = 0
                print('Loading folder: {0}')

    cv2.destroyAllWindows()
    print('Done!')


if __name__ == '__main__':
    main()

