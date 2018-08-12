from tkinter import Tk
from tkinter.filedialog import askdirectory
from classmjpegreader import ClassMjpegReader
from classutils import ClassUtils
from classopenpose import ClassOpenPose
import os
import cv2
import numpy as np
import shutil


def main():
    print('Initializing main function')
    print('Folder selection')

    folder_images = '/home/mauricio/PosesProcessed/folder_images'
    folder_images_draw = '/home/mauricio/PosesProcessed/folder_images_draw'

    Tk().withdraw()

    # Getting options
    init_dir = '/home/mauricio/Videos/Oviedo'
    options = {'initialdir': init_dir}
    dir_name = askdirectory(**options)

    if not dir_name:
        raise Exception('Directory not selected')

    # Create directory if does not exists
    if not os.path.isdir(folder_images):
        os.makedirs(folder_images)

    # Create directory if does not exists
    if not os.path.isdir(folder_images_draw):
        os.makedirs(folder_images_draw)

    # Initializing openpose instance
    instance_pose = ClassOpenPose()

    for root, subdirs, files in os.walk(dir_name):
        for file in files:
            full_path = os.path.join(root, file)
            print('Processing {0}'.format(full_path))

            extension = ClassUtils.get_filename_extension(full_path)

            if extension == '.mjpeg':
                file_info = ClassMjpegReader.process_video(full_path)
            else:
                print('Extension ignored: {0}'.format(extension))
                continue

            # Getting idcam from file
            id_cam = full_path.split('/')[-2]
            print('IdCam: {0}'.format(id_cam))

            for index, info in enumerate(file_info):
                print('Processing {0} of {1} from {2}'.format(index, len(file_info), full_path))
                frame = info[0]
                ticks = info[1]

                image_np = np.frombuffer(frame, dtype="int32")
                image = cv2.imdecode(image_np, cv2.IMREAD_ANYCOLOR)

                arr, image_draw = instance_pose.recognize_image_tuple(image)
                min_score = 0.05

                arr_pass = list()
                for elem in arr:
                    if ClassUtils.check_vector_integrity_part(elem, min_score):
                        arr_pass.append(elem)

                if len(arr_pass) > 0:
                    # Draw rectangles for all candidates
                    for person_arr in arr_pass:
                        pt1, pt2 = ClassUtils.get_rectangle_bounds(person_arr, min_score)
                        cv2.rectangle(image_draw, pt1, pt2, (0, 0, 255), 5)

                    # Overwriting 1
                    full_path_images = os.path.join(folder_images, '{0}_{1}.jpg'.format(ticks, id_cam))
                    print('Writing image {0}'.format(full_path_images))
                    cv2.imwrite(full_path_images, image)

                    # Overwriting 2
                    full_path_draw = os.path.join(folder_images_draw, '{0}_{1}.jpg'.format(ticks, id_cam))
                    print('Writing image {0}'.format(full_path_images))
                    cv2.imwrite(full_path_draw, image_draw)

    print('Done!')


if __name__ == '__main__':
    main()
