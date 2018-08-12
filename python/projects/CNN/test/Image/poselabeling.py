from tkinter import Tk
from tkinter import filedialog
from tkinter import messagebox
from classopenpose import ClassOpenPose
import cv2
import os
import shutil
from classutils import ClassUtils
import json


def main():
    print('Initializing main function')

    # Initializing instances
    instance_pose = ClassOpenPose()

    # Withdrawing tkinter
    Tk().withdraw()

    # Loading folder images
    folder_images = '/home/mauricio/PosesProcessed/folder_images'
    folder_images_draw = '/home/mauricio/PosesProcessed/folder_images_draw'
    pose_base_folder = '/home/mauricio/Pictures/PosesNew'

    # Reading all elements in list
    list_files = sorted(os.listdir(folder_images_draw))

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    index = 0

    while True:
        file = list_files[index]
        full_path_draw = os.path.join(folder_images_draw, file)

        image_draw = cv2.imread(full_path_draw)
        cv2.imshow('main_window', image_draw)

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
            # Ask are you sure question
            result = messagebox.askyesno("Exit", "Are you sure to exit?")
            print(result)

            if result:
                break
        elif key == 115:
            # Check JSON file
            image_full_path = os.path.join(folder_images, file)
            json_path = image_full_path.replace(".jpg", ".json")

            if not os.path.exists(json_path):
                # Generating JSON array
                image = cv2.imread(image_full_path)
                arr = instance_pose.recognize_image(image).tolist()
            else:
                with open(json_path, 'r') as file:
                    arr_str = file.read()
                    arr = json.load(arr_str)

            arr_pass = []
            min_score = 0.05

            # Drawing candidates
            for elem in arr:
                if ClassUtils.check_vector_integrity_part(elem, min_score):
                    arr_pass.append(elem)

            if len(arr_pass) == 0:
                print('Arr pass size equals to zero')
                continue
            elif len(arr_pass) == 1:
                selection_int = 0
                person_arr = arr_pass[0]
                arr_str = json.dumps(person_arr)
            else:
                # Draw rectangles for all candidates
                for index_elem, person_arr in enumerate(arr_pass):
                    pt1, pt2 = ClassUtils.get_rectangle_bounds(person_arr, min_score)
                    cv2.rectangle(image_draw, pt1, pt2, (0, 0, 255), 5)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottom_left_corner = pt2
                    font_scale = 0.6
                    font_color = (255, 255, 255)
                    line_type = 2

                    cv2.putText(image_draw, '{0}'.format(index_elem),
                                bottom_left_corner,
                                font,
                                font_scale,
                                font_color,
                                line_type)

                while True:
                    print('Select image to put')
                    cv2.imshow('main_window', image_draw)
                    wait_key = cv2.waitKey(0)

                    print('Wait Key: {0}'.format(wait_key))
                    selection_int = ClassUtils.key_to_number(wait_key)

                    print('Selection: {0}'.format(selection_int))
                    if 0 <= selection_int < len(arr_pass):
                        break

                person_arr = arr_pass[selection_int]
                arr_str = json.dumps(person_arr)

            # Getting new image using numpy slicing
            pt1, pt2 = ClassUtils.get_rectangle_bounds(person_arr, min_score)
            image_draw = image_draw[pt1[1]:pt2[1], pt1[0]:pt2[0]]

            # Selecting directory after processing image!
            init_dir = '/home/mauricio/Pictures/PosesNew'
            options = {'initialdir': init_dir}
            dir_name = filedialog.askdirectory(**options)

            if not dir_name:
                print('Directory not selected')
            else:
                new_name = ClassUtils.get_filename_no_extension(file)
                new_name = "{0}_{1}.{2}".format(new_name, selection_int, 'jpg')

                new_image_path = os.path.join(dir_name, new_name)
                new_json_path = os.path.join(dir_name, new_name.replace('.jpg', '.json'))

                cv2.imwrite(new_image_path, image_draw)
                with open(new_json_path, 'w') as file:
                    file.write(arr_str)

                print('File copied from {0} to {1}'.format(full_path_draw, new_image_path))
                index += 1
                if index == len(list_files):
                    index = len(list_files) - 1

    cv2.destroyAllWindows()
    print('Done!')


if __name__ == '__main__':
    main()
