"""
Video visualizer
Code inspired in stackoverflow response
https://stackoverflow.com/questions/17987598/how-can-i-use-imshow-to-display-multiple-images-in-multiple-windows
"""
from classpeoplereid import ClassPeopleReId
import cv2
import numpy as np
import math
from datetime import datetime
from datetime import timedelta
from classutils import ClassUtils
from classmjpegdate import ClassMjpegDate
from classmjpegconverter import ClassMjpegConverter
from classdescriptors import ClassDescriptors
from distutils.dir_util import copy_tree
from tkinter import filedialog
from tkinter import Tk
import tkinter as tk
import shutil
import os
import json
from typing import List
from sys import platform
import subprocess


game_period_ms = 500
min_score = 0.05
last_upper = [0, 0, 0]
last_lower = [0, 0, 0]
list_people = list()
list_people_reid = list()
list_candidates = list()
count_person = 0
resize_factor = 1.5
image_width = 0
image_height = 0
saving_activity = False
is_playing = True
forward_until_person = False
saving_person_guid = ''

# Default camera values
list_cams = [419, 420, 421, 428, 429, 430]
date_init = datetime(2018, 2, 24, 14, 15, 0)
date_end = datetime(2018, 2, 24, 15, 15, 0)

SAVE_IMG_BASE_FOLDER = '/home/mauricio/Pictures/BTF/Examples'
CANDIDATES_FOLDER = '/home/mauricio/Pictures/BTF/Candidates'

base_path = '/home/mauricio/Videos/Oviedo'

# Loading dummy frame
with open(ClassUtils.no_img_path, 'rb') as file:
    dummy_frame = file.read()

dummy_frame_cv = cv2.imread(ClassUtils.no_img_path, cv2.IMREAD_COLOR)
option = '0'


def main():
    global option, list_cams, date_init, date_end
    print('Initializing main function')

    option = input('Selecting list cam option: ')
    if option == '1':
        list_cams = [419, 420, 421, 428, 429, 430]
        date_init = datetime(2018, 2, 24, 14, 15, 0)
        date_end = datetime(2018, 2, 24, 15, 15, 0)
    elif option == '2':
        list_cams = [419, 420, 421, 428, 429, 430]
        date_init = datetime(2018, 7, 27, 15, 0, 0)
        date_end = datetime(2018, 7, 27, 15, 14, 59)
    elif option == '3':
        list_cams = [419, 420]
        date_init = datetime(2018, 9, 4, 3, 0, 0)
        date_end = datetime(2018, 9, 4, 3, 15, 0)
    elif option == '4':
        list_cams = [419, 420, 421]
        date_init = datetime(2018, 9, 3, 12, 45, 0)
        date_end = datetime(2018, 9, 3, 12, 59, 59)
    elif option == '5':
        list_cams = [419, 420, 421]
        date_init = datetime(2018, 9, 3, 12, 45, 0)
        date_end = datetime(2018, 9, 3, 12, 59, 59)
    elif option == '6':
        list_cams = [597, 598, 599, 605, 606, 607]
        date_init = datetime(2018, 9, 10, 7, 15, 0)
        date_end = datetime(2018, 9, 10, 7, 30, 0)
    elif option == '7':
        list_cams = [413, 414, 415, 416, 417, 418]
        date_init = datetime(2018, 9, 10, 13, 00, 0)
        date_end = datetime(2018, 9, 10, 13, 15, 0)
    elif option == '8':
        list_cams = [1066, 1067, 1068, 1069, 1070]
        date_init = datetime(2018, 7, 27, 15, 15, 0)
        date_end = datetime(2018, 7, 27, 15, 44, 59)
    elif option == '9':
        list_cams = [419, 420, 421, 428, 429]
        date_init = datetime(2018, 7, 27, 15, 15, 0)
        date_end = datetime(2018, 7, 27, 15, 29, 59)
    elif option == '10':
        list_cams = [419, 420, 421, 428, 429, 430]
        date_init = datetime(2018, 9, 29, 12, 30, 0)
        date_end = datetime(2018, 9, 29, 12, 59, 59)
    elif option == '11':
        list_cams = [419, 420, 421, 428, 429, 430]
        date_init = datetime(2018, 9, 28, 9, 0, 0)
        date_end = datetime(2018, 9, 28, 9, 59, 59)
    elif option == '12':
        list_cams = [900]
        date_init = datetime(2017, 11, 3, 11, 0, 0)
        date_end = datetime(2017, 11, 3, 11, 14, 59)
    elif option == '13':
        list_cams = [99]
        date_init = datetime(2018, 10, 8, 17, 30, 0)
        date_end = datetime(2018, 10, 8, 17, 59, 59)
    elif option == '14':
        list_cams = [98]
        date_init = datetime(2018, 10, 10, 21, 30, 0)
        date_end = datetime(2018, 10, 10, 21, 59, 59)
    elif option == '15':
        list_cams = [411]
        date_init = datetime(2018, 9, 28, 10, 0, 0)
        date_end = datetime(2018, 9, 28, 10, 29, 59)
    elif option == '16':
        list_cams = [419, 420, 421, 428, 429, 430]
        date_init = datetime(2018, 10, 30, 8, 30, 0)
        date_end = datetime(2018, 10, 30, 12, 59, 59)
    else:
        raise Exception('Option not recognized')

    select_options()


def mouse_callback(event, x_image, y_image, flags, param):
    global list_cams

    if event == cv2.EVENT_LBUTTONDOWN:
        index_x = 0
        index_y = 0

        new_x = x_image
        new_y = y_image

        # Iterate until new_x below image index
        while new_x >= image_width:
            new_x -= image_width
            index_x += 1

        while new_y >= image_height:
            new_y -= image_height
            index_y += 1

        new_x = new_x * resize_factor
        new_y = new_y * resize_factor

        index_img = index_x + index_y * 3

        if index_img >= len(list_cams):
            print('Index image is outside index cams: {0}'.format(len(list_cams)))
        else:
            id_cam = str(list_cams[index_img])
            if not ClassUtils.cam_calib_exists(id_cam):
                print('Cam calibration invalid for id cam {0}'.format(id_cam))
            else:
                calib_params = ClassUtils.load_cam_calib_params(id_cam)

                center = np.array(calib_params['centerPoints'])
                angle_deg = calib_params['angleDegrees']
                homo_mat = np.array(calib_params['homographyMat'])
                projected = ClassUtils.project_points_angle(homo_mat,
                                                            np.asanyarray([new_x,
                                                                           new_y],
                                                                          dtype=np.float),
                                                            center, angle_deg)

                print('Point{0} - IdCam: {1} - Projected: {2}'.format([new_x, new_y], id_cam, projected))


def select_options():
    Tk().withdraw()

    print('Initializing main function')
    selection = input('Press 1 to generate files, 2 to check files, 3 to perform debugging, 4 to get poses - '
                      '5 to debug pose generation: ')

    if selection == '1':
        generate_files()
    elif selection == '2':
        check_files()
    elif selection == '3':
        debug_reid()
    elif selection == '4':
        get_poses()
    elif selection == '5':
        debug_pose_generation()
    else:
        raise Exception('Selection not identified')


def generate_files():
    global list_people, list_people_reid

    result = input('Delete CNN Folder? (y/n): ')
    if result == 'y' or result == 'Y':
        print('Removing CNN folder and CNN partial folder')

        folder_cnn = os.path.join(ClassUtils.cnn_folder, option)
        if os.path.exists(folder_cnn):
            shutil.rmtree(folder_cnn)

        folder_cnn_mov = os.path.join(ClassUtils.cnn_partial_folder_mov, option)
        if os.path.exists(folder_cnn_mov):
            shutil.rmtree(folder_cnn_mov)

        folder_cnn_no_mov = os.path.join(ClassUtils.cnn_partial_folder_no_mov, option)
        if os.path.exists(folder_cnn_no_mov):
            shutil.rmtree(folder_cnn_no_mov)

    print('Initializing generate files')

    date_video = date_init
    list_readers = list()
    list_paths = list()

    for i in range(len(list_cams)):
        list_readers.append(ClassMjpegDate(list_cams[i]))

    while date_video < date_end:
        print(date_video)

        # Getting list frames first
        frame_info_list = list()
        for i in range(len(list_cams)):
            frame_info = list_readers[i].load_frame(date_video)

            # Generate a guid for every pose
            # Generating blank person guids
            # Reid purposes
            for param in frame_info[2]['params']:
                param['ticks'] = ClassUtils.datetime_to_ticks(date_video)

            frame_info_list.append(frame_info)

        # Perform re-identification module
        process_reid(frame_info_list, date_video)

        # Write in image
        for i in range(len(list_cams)):
            date_file = ClassUtils.get_date_file(date_video)
            video_path = ClassUtils.load_path_by_date(date_file, str(list_cams[i]), '.mjpegx')

            if video_path not in list_paths:
                list_paths.append(video_path)

        # Next
        date_video = date_video + timedelta(milliseconds=game_period_ms)

    # Add current people elems into list
    for person in list_people:
        list_people_reid.append(person)

    # Delete people with 3 or less true elements
    list_people_to_remove = list()
    for person in list_people_reid:
        count = 0
        for pose in person.list_poses:
            pose_guid = pose['poseGuid']
            if pose_guid != '':
                count += 1

        if count <= 3:
            list_people_to_remove.append(person)

    for person in list_people_to_remove:
        list_people_reid.remove(person)

    # Update mjpegx using list_people_reid info
    for path in list_paths:
        print('Converting video: {0}'.format(path))
        ClassMjpegConverter.convert_video_mjpegx_reid(path, list_people_reid, save_img=True, option=option)

    # Update person element into folder
    for person in list_people_reid:
        person_guid = person.person_guid
        list_poses = person.list_poses
        save_list_poses(person_guid, list_poses)

    # Processing partial list for re-identification
    process_list_partial()


def get_base_folder_video(person_guid, moving=False, is_partial=False, count=0):
    if not is_partial:
        new_guid = person_guid
        base_folder = os.path.join(ClassUtils.cnn_folder, option)
        color_back = (255, 255, 255)
    else:
        new_guid = '{0}_{1}'.format(person_guid, count)
        if moving:
            color_back = (255, 200, 220)
            base_folder = os.path.join(ClassUtils.cnn_partial_folder_mov, option)
        else:
            color_back = (255, 255, 255)
            base_folder = os.path.join(ClassUtils.cnn_partial_folder_no_mov, option)

    path_folder = os.path.join(base_folder, new_guid)
    return path_folder, color_back, new_guid


def save_list_poses(person_guid, list_poses, save_example_global=False, moving=False, is_partial=False, count=0):
    path_folder, color_back, new_guid = get_base_folder_video(person_guid, moving, is_partial, count)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    filename = os.path.join(path_folder, '{0}.json'.format(new_guid))
    with open(filename, 'w') as f:
        f.write(json.dumps({
            'personGuid': new_guid,
            'listPoses': list_poses,
            'moving': moving
        }, indent=2))

    if save_example_global:
        for pose in list_poses:
            save_pose_global(path_folder, pose, color_back=color_back)

        if is_partial:
            # Debugging purposes
            # Save pose partial - Debugging purposes
            new_base_folder = os.path.join(ClassUtils.cnn_folder, option, person_guid, 'partial')
            new_path_folder = os.path.join(new_base_folder, new_guid)
            for pose in list_poses:
                save_pose_global(new_path_folder, pose, color_back=color_back)

            # Saving original pose folder
            ori_folder = os.path.join(path_folder, 'ori')

            print('Copy files from dir {0} to dir {1}'.format(path_folder, ori_folder))
            copy_tree(path_folder, ori_folder)

    # Done!
    # Return path folder for reference purposes
    return path_folder


def save_pose_partial(person_guid, count, list_poses_partial, moving):
    print('Saving pose partial for guid: {0} count: {1}'.format(person_guid, count))
    save_list_poses(person_guid, list_poses_partial, save_example_global=True,
                    moving=moving, is_partial=True, count=count)


def process_list_partial():
    # Create partial list of elems
    for person in list_people_reid:
        moving = True

        index = 0
        num_poses_future = 6
        min_distance_x = 80
        min_distance_y = 160
        list_poses_partial = list()
        count = 0

        while index < len(person.list_poses):
            pose = person.list_poses[index]
            remaining = len(person.list_poses) - index - 1

            pos = pose['globalPosition']

            if moving:
                valid = True
                if remaining >= num_poses_future:
                    for i in range(index + 1, index + num_poses_future):
                        pos_future = person.list_poses[index + num_poses_future]['globalPosition']

                        distance_x = math.fabs(pos[0] - pos_future[0])
                        distance_y = math.fabs(pos[1] - pos_future[1])
                        if distance_x < min_distance_x and distance_y < min_distance_y:
                            valid = False
                            break

                if not valid:
                    # Saving current list and create a new one
                    if len(list_poses_partial) != 0:
                        save_pose_partial(person.person_guid, count, list_poses_partial, moving)
                        list_poses_partial.clear()
                        count += 1

                    moving = False

                list_poses_partial.append(pose)
            else:
                # Generating elements into list
                valid = True
                if remaining >= num_poses_future:
                    for i in range(index + 1, index + num_poses_future):
                        pos_future = person.list_poses[index + num_poses_future]['globalPosition']

                        distance_x = math.fabs(pos[0] - pos_future[0])
                        distance_y = math.fabs(pos[1] - pos_future[1])

                        # Inverse
                        if distance_x >= min_distance_x or distance_y >= min_distance_y:
                            valid = False
                            break

                list_poses_partial.append(pose)
                if not valid:
                    # New pose list
                    for i in range(num_poses_future):
                        list_poses_partial.append(person.list_poses[index + i])

                    save_pose_partial(person.person_guid, count, list_poses_partial, moving)
                    list_poses_partial.clear()
                    count += 1

                    index += num_poses_future
                    moving = True

            # Iterator!
            index += 1

        if len(list_poses_partial) != 0:
            # Saving elements in partial format
            save_pose_partial(person.person_guid, count, list_poses_partial, moving)
            list_poses_partial.clear()
            count += 1

    print('Done!')


def save_pose_global(path_folder, pose, color_back=(255, 255, 255)):
    transformed_points = pose['transformedPoints']
    global_position = pose['globalPosition']
    key_pose = pose['keyPose']
    ticks = pose['ticks']

    re_scale_factor = 100
    scaled_points = ClassDescriptors.re_scale_pose_factor(transformed_points, re_scale_factor, min_score)

    person_array = list()
    person_array.append([0, 0, 0])
    person_array.append([0, 0, 1])
    for point in scaled_points:
        person_array.append([point[0], point[1], 1])
    for _ in range(10):
        person_array.append([0, 0, 0])

    local_position = ClassDescriptors.get_local_position_point(person_array, min_score, key_pose)

    # Draw global position
    min_x = -200
    max_x = 1000

    min_y = -900
    max_y = 1200

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    width_plane = 1000
    height_plane = 500

    width_rect = 10

    img_plane = np.zeros((height_plane, width_plane, 3), np.uint8)

    new_person_array = list()

    # Height plane must be set into list
    pt_plane_x = int((global_position[0] - min_x) * width_plane / delta_x)
    pt_plane_y = height_plane - int((global_position[1] - min_y) * height_plane / delta_y)

    pt1 = (pt_plane_x - width_rect, pt_plane_y - width_rect)
    pt2 = (pt_plane_x + width_rect, pt_plane_y + width_rect)

    delta_x_pos = pt_plane_x - local_position[0]
    delta_y_pos = pt_plane_y - local_position[1]

    # Blank image
    img_plane[:, :] = color_back

    # Transform local vector coordinates
    for point in person_array:
        if ClassUtils.check_point_integrity(point, min_score):
            new_person_array.append([point[0] + delta_x_pos, point[1] + delta_y_pos, 1])
        else:
            new_person_array.append(point)

    # Draw pose
    ClassDescriptors.draw_pose(img_plane, new_person_array, min_score, key_pose)

    # Fill rectangle
    cv2.rectangle(img_plane, pt1, pt2, (255, 0, 255), thickness=-1)

    # Save global position info
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 0, 0)
    line_type = 2

    cv2.putText(img_plane, '({0:.2f}, {1:.2f})'.format(global_position[0], global_position[1]), pt2,
                font, font_scale, font_color, line_type)

    # Save image with guid
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    filename = os.path.join(path_folder, '{0}_g.jpg'.format(ticks))
    print('Saving image {0}'.format(filename))
    cv2.imwrite(filename, img_plane)


def check_files():
    global list_people, is_playing, forward_until_person
    print('Initializing check files')

    date_video = date_init
    list_readers_mjpegx = list()
    for i in range(len(list_cams)):
        list_readers_mjpegx.append(ClassMjpegDate(list_cams[i], extension='.mjpegx'))

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)

    is_playing = True
    forward_until_person = False
    play_factor = 1
    while date_video < date_end:
        frame_info_list = list()

        for i in range(len(list_cams)):
            frame_info = list_readers_mjpegx[i].load_frame(date_video)
            frame_info_list.append(frame_info)

        if is_playing:
            print(date_video)

            # Loading list people until now
            list_people = ClassPeopleReId.load_list_people(frame_info_list, date_video)

            # Draw poses and vectors
            draw_images(frame_info_list)

        # Waiting for pressed key
        if not forward_until_person:
            key = cv2.waitKey(game_period_ms)
        else:
            count = check_people_available(frame_info_list)
            if count == 0:
                key = cv2.waitKey(1)
            else:
                # Wait key and stops playing
                key = cv2.waitKey(game_period_ms)
                is_playing = False
                forward_until_person = False

        if key != -1:
            print('KeyPressed: {0}'.format(key))

        if key == 27:
            # Esc
            break

        # Process key commands
        date_video = process_date_video(key, date_video)
        play_factor = process_play_factor(key, play_factor)
        process_save_image(key, date_video, frame_info_list)
        process_is_playing(key)
        process_forward(key)

    cv2.destroyAllWindows()
    print('Done!')


def process_forward(key):
    global forward_until_person
    if key == 102:
        # Forward until key
        if forward_until_person:
            print('Disabling forward_until_person')
            forward_until_person = False
        else:
            print('Enabling forward_until_person')
            forward_until_person = True


def check_people_available(frame_info_list):
    print('Check people available')
    count_vectors = 0

    for frame_info in frame_info_list:
        json_dict = frame_info[2]

        params = json_dict['params']
        for param in params:
            if param['integrity']:
                count_vectors += 1

    return count_vectors


def process_date_video(key, date_video):
    global list_people, is_playing
    new_date_video = date_video
    if is_playing:
        if key == 52:
            # Left arrow
            # Seconds
            list_people.clear()
            new_date_video = date_video - timedelta(milliseconds=1000 * 3)
            if new_date_video < date_init:
                new_date_video = date_init
        elif key == 54:
            # Right arrow
            # Seconds
            list_people.clear()
            new_date_video = date_video + timedelta(milliseconds=1000 * 3)
        elif key == 49:
            # Arrow 1
            # Minutes
            list_people.clear()
            new_date_video = date_video - timedelta(milliseconds=1000 * 60)
            if new_date_video < date_init:
                new_date_video = date_init
        elif key == 51:
            # Arrow 3
            # Minutes
            list_people.clear()
            new_date_video = date_video + timedelta(milliseconds=1000 * 60)
        elif key == 55:
            # Arrow 7
            # Middle
            list_people.clear()
            new_date_video = date_video - timedelta(milliseconds=1000 * 20)
            if new_date_video < date_init:
                new_date_video = date_init
        elif key == 57:
            list_people.clear()
            new_date_video = date_video + timedelta(milliseconds=1000 * 20)
        else:
            # Normal game play
            new_date_video = date_video + timedelta(milliseconds=game_period_ms)

    return new_date_video


def process_save_image(key, date_video: datetime, frame_info_list: list):
    global is_playing

    # Function to save silhouettes or poses
    if is_playing:
        # Ignore!
        return

    # S key pressed
    save_key_number = 115
    if key == save_key_number:
        print('Select image to save')
        key = cv2.waitKey()
        key_number = ClassUtils.cv_key_to_number(key)

        print('Key number: {0}'.format(key_number))

        if key_number == -1:
            print('Invalid key number')
            return

        if key_number >= len(frame_info_list):
            print('key_number greater than frame_info_list: {0}'.format(len(frame_info_list)))
            return

        frame_info = frame_info_list[key_number]
        image = frame_info[0]

        options = {
            'initialdir': '/home/mauricio/Pictures',
            'defaultextension': '.jpg'
        }

        filename = filedialog.asksaveasfilename(**options)
        if filename is None:
            print('Operation canceled')
        else:
            print('Filename: {0}'.format(filename))

            with open(filename, 'wb') as f:
                f.write(image)

            print('Image saved with name: {0}'.format(filename))
    else:
        # Not image key pressed
        process_save_pose(key, date_video)


def process_save_pose(key, date_video):
    global list_candidates, label, ok_params

    key_number = ClassUtils.cv_key_to_number(key)

    if key_number == -1:
        # Ignore!
        return

    print('Selected image: {0}'.format(key_number))
    filter_list = list(filter(lambda item: item['index'] == key_number, list_candidates))

    if len(filter_list) == 0:
        print('Invalid len for list')
    else:
        # Saving index from image
        master = Tk()
        tk.Label(master, text="label").grid(row=0)
        e1 = tk.Entry(master)
        e1.grid(row=0, column=1)
        label = ''
        ok_params = False

        def read_params():
            global label, ok_params

            label = int(e1.get())
            ok_params = True
            master.quit()

        tk.Button(master, text='Quit', command=master.quit).grid(row=1, column=0, pady=4)
        tk.Button(master, text='OK', command=read_params).grid(row=1, column=1, pady=4)
        # Handle closing event window
        master.protocol("WM_DELETE_WINDOW", master.quit)

        master.mainloop()

        # Destroying window
        master.destroy()

        if not ok_params:
            print('Deleting element')
            return

        # Opening save file dialog
        init_dir = SAVE_IMG_BASE_FOLDER
        options = {
            'initialdir': init_dir,
        }

        folder = filedialog.askdirectory(**options)

        if folder is None:
            print('Folder not selected')
        else:
            candidate = list_candidates[key_number]
            cam_number = candidate['camNumber']
            param = candidate['param']
            param['index'] = candidate['index']
            param['label'] = label

            ticks = ClassUtils.datetime_to_ticks(date_video)
            filename = '{0}-{1}-{2}.jpg'.format(label, cam_number, ticks)
            full_path = os.path.join(folder, filename)
            full_path_json = full_path.replace('.jpg', '.json')

            img_cv = candidate['imageCv']

            print('Writing file cv {0}'.format(full_path))
            cv2.imwrite(full_path, img_cv)

            print('Writing file json {0}'.format(full_path_json))
            with open(full_path_json, 'w') as f:
                f.write(json.dumps(param, indent=2))

            # Saving candidate if not exists - Referencing purposes
            if not os.path.exists(CANDIDATES_FOLDER):
                os.makedirs(CANDIDATES_FOLDER)

            file_candidate = os.path.join(CANDIDATES_FOLDER, '{0}.jpg'.format(label))
            if not os.path.exists(file_candidate):
                cv2.imwrite(file_candidate, img_cv)
                candidate_path_json = file_candidate.replace('.jpg', '.json')
                with open(candidate_path_json, 'w') as f:
                    f.write(json.dumps(param, indent=2))

    # Done writing elements


def process_init_save_activity(key):
    global saving_activity, list_people, saving_person_guid, is_playing

    if key != 97:
        # Ignore if key is not a
        return

    if is_playing:
        print('Video is playing - Must be stopped!')
        return

    # Select person
    print('Select person in window')
    key = cv2.waitKey()

    key_number = ClassUtils.cv_key_to_number(key)
    print('Key number: {0}'.format(key_number))

    if key_number == -1:
        print('Invalid key number')
        return

    if key_number >= len(list_people):
        print('key_number greater than list_people len: {0}'.format(len(list_people)))
        return

    print('Key number selected: {0}'.format(key_number))
    person = list_people[key_number]

    saving_person_guid = person.person_guid
    saving_activity = True
    print('Person guid is: {0}'.format(saving_person_guid))
    print('Done!')


def process_finish_save_activity(key):
    global saving_activity, saving_person_guid, list_people, list_people_reid, is_playing

    if key != 122:
        # Ignore if key is not z
        return

    if not saving_activity:
        print('Saving activity is not active!')
        return

    saving_guid_activity()
    return


def saving_guid_activity():
    global saving_activity, saving_person_guid, list_people, list_people_reid, is_playing

    print('Saving person guid: {0}'.format(saving_person_guid))

    # Finding person guid in list people
    list_people_reid.clear()
    for person in list_people:
        if person.person_guid == saving_person_guid:
            list_people_reid.append(person)
            break

    if len(list_people_reid) == 0:
        print('Cant find person with guid: {0}'.format(saving_person_guid))
        return

    # Saving poses!
    path_folder = ''
    for person in list_people_reid:
        person_guid = person.person_guid
        list_poses = person.list_poses
        save_list_poses(person_guid, list_poses, save_example_global=True)
        path_folder, _, _ = get_base_folder_video(person.person_guid)

    # Processing partial list for re-identification
    process_list_partial()

    # Reset flags and open window in explorer
    saving_activity = False
    saving_person_guid = ''

    # Showing in explorer
    if platform == 'win32':
        subprocess.Popen(['explorer', path_folder])
    else:
        subprocess.Popen(['xdg-open', path_folder])

    # Clearing to avoid action overlapping
    list_people.clear()

    # Stop playing
    is_playing = False


def process_is_playing(key):
    global is_playing
    if key == 53:
        # Don't change date video, but avoid processing
        if is_playing:
            is_playing = False
        else:
            is_playing = True


def process_play_factor(key, play_factor):
    new_play_factor = play_factor
    if key == 190:
        # Change playing factor
        new_play_factor = 1
        print('Set play_factor to {0}'.format(new_play_factor))
    elif key == 191:
        play_factor = 2
        print('Set play_factor to {0}'.format(new_play_factor))
    elif key == 193:
        play_factor = 4
        print('Set play_factor to {0}'.format(new_play_factor))

    return new_play_factor


def debug_reid():
    global is_playing, forward_until_person
    print('Initializing main function')

    date_video = date_init
    list_readers = []
    for i in range(len(list_cams)):
        list_readers.append(ClassMjpegDate(list_cams[i]))

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    cv2.setMouseCallback('main_window', mouse_callback)

    is_playing = True
    forward_until_person = False
    play_factor = 1
    while date_video < date_end:
        frame_info_list = list()
        ticks_video = ClassUtils.datetime_to_ticks(date_video)

        # Generate frame_info_list always
        for i in range(len(list_cams)):
            frame_info = list_readers[i].load_frame(date_video)

            # Generate a guid for every pose
            # Generating blank person guids
            # Reid purposes
            for param in frame_info[2]['params']:
                param['ticks'] = ticks_video

            frame_info_list.append(frame_info)

        if is_playing:
            print('DateVideo: {0} - ticks: {1}'.format(date_video, ticks_video))

            # Getting list frames first
            # Waiting for pressed key
            if forward_until_person:
                count = check_people_available(frame_info_list)
                if count != 0:
                    # Wait and stops
                    forward_until_person = False
                    is_playing = False

            # Avoid reid and drawing
            if not forward_until_person:
                # Perform re-identification module
                process_reid(frame_info_list, date_video)

                # Draw poses and vectoris
                draw_images(frame_info_list)

        # Fast selection to forward
        if not forward_until_person:
            key = cv2.waitKey(int(game_period_ms / 2)) # Fast Test
        else:
            key = cv2.waitKey(1)

        if key != -1:
            print('KeyPressed: {0}'.format(key))

        if key == 27:
            # Esc
            break

        # Process elems
        date_video = process_date_video(key, date_video)
        play_factor = process_play_factor(key, play_factor)
        process_save_image(key, date_video, frame_info_list)
        process_is_playing(key)
        process_forward(key)
        process_init_save_activity(key)
        process_finish_save_activity(key)

    cv2.destroyAllWindows()
    print('Done!')


def process_reid(frame_info_list, date_ref: datetime):
    global list_people, list_people_reid, saving_activity, saving_person_guid
    list_new_people: List[ClassPeopleReId] = list()

    # Load list people
    for frame_info in frame_info_list:
        list_new_people += ClassPeopleReId.load_people_from_frame_info(frame_info, date_ref)

    # Function to merge people with overlapped cameras
    def merge_list():
        global done
        done = True
        for i in range(len(list_new_people) - 1):
            person1 = list_new_people[i]
            list_candidate_merge = []
            for j in range(i + 1, len(list_new_people)):
                person2 = list_new_people[j]

                # To do merging
                # People must be in different cameras
                if person1.cam_number != person2.cam_number:
                    return_data = ClassPeopleReId.get_people_diff(person1, person2)

                    if return_data['distance'] <= 120 and return_data['diffKMeans'] < 20:
                        list_candidate_merge.append({
                            'person': person2,
                            'data': return_data
                        })

            if len(list_candidate_merge) > 0:
                # Iterate over elements to get the elem with the min distance
                min_diff = -1
                person_merge = None

                for candidate in list_candidate_merge:
                    return_data = candidate['data']
                    total_diff = return_data['diffKMeans']

                    if min_diff == -1 or total_diff < min_diff:
                        min_diff = total_diff
                        person_merge = candidate['person']

                # Checking what candidate hast the most valid points
                print('Merging {0} {1}'.format(person1.global_pos, person_merge.global_pos))
                valid1 = ClassUtils.check_valid_vector_points(person1.vectors, min_score)
                valid2 = ClassUtils.check_valid_vector_points(person_merge.vectors, min_score)

                if valid1 >= valid2:
                    list_new_people.remove(person_merge)
                else:
                    list_new_people.remove(person1)

                done = False
                break
            else:
                continue

        return done

    # Do cycle until there are no more vectors to merge
    while True:
        res_merge = merge_list()
        if res_merge:
            break

    # Compare vectors with previous elements
    updated_people: List[ClassPeopleReId] = list()

    # Re identification part
    # Find candidates
    # Select less score
    for person in list_new_people:
        list_candid: List[ClassPeopleReId] = list()
        for person_last in list_people:
            if person_last not in updated_people:
                score = ClassPeopleReId.compare_people(person, person_last)
                print('Score: {0} - {1} - {2} Color: {3}'.format(person.global_pos, person_last.global_pos, score
                                                                 , person_last.get_rgb_color_str_int()))

                upper, lower, dis = ClassPeopleReId.compare_people_items(person, person_last)
                print('Upper: {0:.4f}, Lower: {1:.4f}, Distance: {2:.4f}'.format(upper, lower, dis))

                if score <= 0.5:
                    list_candid.append(person_last)

        # Evaluate candidates for selected elements
        if len(list_candid) > 0:
            minimum_score = -1
            selected_person = None

            for person_candidate in list_candid:
                score = ClassPeopleReId.compare_people(person, person_candidate)
                if minimum_score == -1 or score < minimum_score:
                    minimum_score = score
                    selected_person = person_candidate

            print('updating {0} with {1}'.format(person.global_pos, selected_person.global_pos))

            # If there is a person with only_pos flag
            # Take key pose from last position - Use position only
            if person.only_pos:
                person.person_param['keyPose'] = selected_person.person_param['keyPose']
                person.person_param['probability'] = selected_person.person_param['probability']

            # If there are gaps in pose list
            # An element must be added
            counter = selected_person.update_counter
            if counter != 0:
                new_position = [person.global_pos[0], person.global_pos[1]]
                old_position = [selected_person.global_pos[0], selected_person.global_pos[1]]

                delta_x = (new_position[0] - old_position[0]) / (counter + 1)
                delta_y = (new_position[1] - old_position[1]) / (counter + 1)

                delta_date = (date_ref - selected_person.last_date).total_seconds() / (counter + 1)
                current_pos = [old_position[0], old_position[1]]
                date = selected_person.last_date

                # Guess positions - Hand and half
                # Iterate over list
                for i in range(counter):
                    if i >= counter / 2:
                        selected_person.person_param['transformedPoints'] = person.person_param['transformedPoints']

                    current_pos[0] += delta_x
                    current_pos[1] += delta_y
                    date += timedelta(seconds=delta_date)

                    # Blank pose guid - Avoid conflicts with mjpegx frames
                    selected_person.person_param['globalPosition'] = [current_pos[0], current_pos[1], 1]
                    selected_person.person_param['poseGuid'] = ''
                    selected_person.update_values_from_person(selected_person, date)

            # Update last frame to person
            selected_person.update_values_from_person(person, date_ref)
            updated_people.append(selected_person)
        else:
            # No person candidate - Must create one in list
            print('Creating new person {0}'.format(person.global_pos))
            list_people.append(person)
            updated_people.append(person)

    # Add not updated flag to people not in updated_people
    for person in list_people:
        if person not in updated_people:
            person.set_not_updated()

    # Remove old elements from list
    # No more than 5 seconds
    remove_people: List[ClassPeopleReId] = list()
    for person in list_people:
        delta = date_ref - person.last_date

        if delta.seconds > 5:
            remove_people.append(person)

    for person in remove_people:
        # If people is into list
        # Remove people
        if person.person_guid == saving_person_guid and saving_activity:
            # Person gets auto removed here!
            # Is playing also gets stopped!
            saving_guid_activity()
        else:
            # Remove and save elements into reid list
            list_people.remove(person)

        if len(person.list_poses) >= 4:
            list_people_reid.append(person)
        else:
            print('List poses less than 4 for person with guid: {0}'.format(person.person_guid))

    # Done


def draw_images(frame_info_list):
    global list_candidates, count_person

    list_candidates.clear()
    count_person = 0

    # Creating images and drawing vectors
    list_images = list()
    for frame_info in frame_info_list:
        dict_frame = frame_info[2]
        cam_number = dict_frame['camNumber']

        if frame_info is None:
            image_arr = np.frombuffer(dummy_frame, dtype=np.uint8)
            image_cv = cv2.imdecode(image_arr, cv2.IMREAD_ANYCOLOR)
        else:
            image_arr = np.frombuffer(frame_info[0], dtype=np.uint8)
            image_cv = cv2.imdecode(image_arr, cv2.IMREAD_ANYCOLOR)
            image_cv_ori = cv2.imdecode(image_arr, cv2.IMREAD_ANYCOLOR)
            draw_vectors(image_cv, image_cv_ori, frame_info)

        list_images.append({
            'imageCv': image_cv,
            'camNumber': cam_number
        })

    # Drawing person info
    draw_people(list_images)

    # Show list images in window
    show_images(list_images)


def draw_vectors(image: np.ndarray, image_ori: np.ndarray, frame_info):
    global count_person, list_candidates

    # Drawing people into image
    dict_frame = frame_info[2]
    cam_number = dict_frame['camNumber']
    params = dict_frame['params']
    found = dict_frame['found']

    # Draw all poses
    for i, param in enumerate(params):
        vectors = param['vectors']
        key_pose = param['keyPose']

        if ClassUtils.check_vector_integrity_pos(vectors, min_score) and found:
            # Draw valid poses in red and put number
            pt1, pt2 = ClassUtils.get_rectangle_bounds(vectors, min_score)
            # Avoid drawing red
            # cv2.rectangle(image, pt1, pt2, (0, 0, 255), 5)

            # Draw info vector
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_color = (0, 0, 255)
            line_type = 2

            cv2.putText(image, '{0}'.format(count_person), pt2, font, font_scale, font_color, line_type)

            list_candidates.append({
                'vector': vectors,
                'index': count_person,
                'imageCv': image_ori,
                'param': param,
                'camNumber': cam_number
            })

            count_person = count_person + 1

        ClassDescriptors.draw_pose(image, vectors, min_score, key_pose)

    # Done drawing vectors


def draw_people(list_images):
    global list_people

    for idx, person in enumerate(list_people):
        image_cv = None
        cam_number = person.cam_number

        for image in list_images:
            if cam_number == image['camNumber']:
                image_cv = image['imageCv']
                break

        if image_cv is None:
            raise Exception('Cant find image for camNumber: {0}'.format(cam_number))

        global last_upper, last_lower

        # Draw elements into image
        pt1, pt2 = ClassUtils.get_rectangle_bounds(person.vectors, min_score)
        cv2.rectangle(image_cv, pt1, pt2, person.get_bgr_color(), 5)

        # Draw info vector
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        line_type = 2

        delta_y = 30
        pos_txt = (pt1[0], pt1[1])

        # Add global position
        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'gpX {0:.0f}'.format(person.global_pos[0]), pos_txt,
                    font, font_scale, font_color, line_type)

        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'gpY {0:.0f}'.format(person.global_pos[1]), pos_txt,
                    font, font_scale, font_color, line_type)

        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'lpX {0:.0f}'.format(person.local_pos[0]), pos_txt,
                    font, font_scale, font_color, line_type)

        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'lpY {0:.0f}'.format(person.local_pos[1]), pos_txt,
                    font, font_scale, font_color, line_type)

        # Upper
        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'uR {0:.0f}'.format(person.color_upper[0]), pos_txt,
                    font, font_scale, font_color, line_type)
        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'uG {0:.0f}'.format(person.color_upper[1]), pos_txt,
                    font, font_scale, font_color, line_type)
        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'uB {0:.0f}'.format(person.color_upper[2]), pos_txt,
                    font, font_scale, font_color, line_type)

        # Lower
        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'lR {0:.0f}'.format(person.color_lower[0]), pos_txt,
                    font, font_scale, font_color, line_type)
        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'lG {0:.0f}'.format(person.color_lower[1]), pos_txt,
                    font, font_scale, font_color, line_type)
        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'lB {0:.0f}'.format(person.color_lower[2]), pos_txt,
                    font, font_scale, font_color, line_type)

        # diff
        diff_colors = ClassUtils.get_color_diff_rgb(person.color_upper, person.color_lower)

        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'DC {0:.0f}'.format(diff_colors), pos_txt,
                    font, font_scale, font_color, line_type)

        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'idx {0:.0f}'.format(idx), pos_txt,
                    font, font_scale, font_color, line_type)

        guid = person.person_guid
        last_guid = guid.split('-')[-1]
        pos_txt = (pos_txt[0], pos_txt[1] + delta_y)
        cv2.putText(image_cv, 'p.guid {0}'.format(last_guid), pos_txt,
                    font, font_scale, font_color, line_type)

        last_upper = person.color_upper
        last_lower = person.color_lower


def show_images(list_images):
    global resize_factor, image_width, image_height

    if len(list_images) == 0:
        raise Exception('Invalid len for list_images')

    base_iter = 6
    if base_iter % 2 != 0:
        raise Exception('Base iter must be even')

    if len(list_images) > base_iter:
        raise Exception('Len images is greate than base_iter: {0}'.format(len((list_images))))

    image_up = None
    image_down = None

    base_width = list_images[0]['imageCv'].shape[1]
    base_height = list_images[0]['imageCv'].shape[0]

    for i in range(base_iter):
        if i >= len(list_images):
            base_image = cv2.resize(dummy_frame_cv, (base_width, base_height))
        else:
            base_image = list_images[i]['imageCv']  # type: np.ndarray
            if base_image.shape[0] != base_height or base_image.shape[1] != base_width:
                raise Exception('Invalid shape for image: {0}'.format(base_image.shape))

        if i < base_iter / 2:
            if image_up is None:
                image_up = base_image
            else:
                image_up = np.hstack((image_up, base_image))
        else:
            if image_down is None:
                image_down = base_image
            else:
                image_down = np.hstack((image_down, base_image))

    result_image = np.vstack((image_up, image_down))

    new_y = int(result_image.shape[0] / resize_factor)
    new_x = int(result_image.shape[1] / resize_factor)

    image_height = int(new_y / 2)
    image_width = int(new_x / (base_iter / 2))

    result_image = cv2.resize(result_image, (new_x, new_y))
    cv2.imshow('main_window', result_image)


def get_poses():
    # The goal is save images with poses
    # Deprecated
    # Use create files instead
    print('Generating poses from videos')
    print('WARNING: Cleaning base folder {0}'.format(ClassUtils.cnn_folder_mov))

    # Cleaning base tree folder
    if os.path.exists(ClassUtils.cnn_folder_mov):
        shutil.rmtree(ClassUtils.cnn_folder_mov)
    os.makedirs(ClassUtils.cnn_folder_mov)

    # Person list
    list_people_all = list()

    date_video = date_init
    list_readers = []
    for i in range(len(list_cams)):
        list_readers.append(ClassMjpegDate(list_cams[i], extension='.mjpegxr'))

    while date_video < date_end:
        print(date_video)

        # Getting list frames first
        frame_info_list = list()

        for i in range(len(list_cams)):
            frame_info = list_readers[i].load_frame(date_video)
            frame_info_list.append(frame_info)

        # Loading list people until now
        list_people_local = ClassPeopleReId.load_list_people(frame_info_list, date_video)

        # Get list people to append elements
        for person_local in list_people_local:
            exists = False

            for list_person in list_people_all:
                # Assume at least one elem
                person = list_person[0]
                if person.get_person_guid() == person_local.get_person_guid():
                    exists = True
                    list_person.append(person_local)
                    break

            if not exists:
                # Append new element
                list_person = list()
                list_person.append(person_local)
                list_people_all.append(list_person)

        date_video = date_video + timedelta(milliseconds=game_period_ms)

    # Generate single file for each person
    for list_person in list_people_all:
        # Assume at least one elem
        person = list_person[0]

        # Generate route based on person guid
        image_folder = os.path.join(ClassUtils.cnn_folder_mov, person.get_person_guid())
        if not os.path.isdir(image_folder):
            os.makedirs(image_folder)

        # List poses
        file_path = os.path.join(image_folder, 'poses.json')

        list_poses = list()
        for person_item in list_person:
            list_poses.append(person_item.person_param)

        print('Writing file: {0}'.format(file_path))
        with open(file_path, 'w') as f:
            f.write(json.dumps(list_poses, indent=2))

    print('Done!')


def debug_pose_generation():
    print('Initializing debug pose generation')

    # Try to draw image pose sequence
    init_dir = ClassUtils.cnn_partial_folder_mov
    options = {'initialdir': init_dir}
    filename = filedialog.askopenfilename(**options)

    if filename is None:
        raise Exception('Filename not selected')

    with open(filename, 'r') as f:
        json_dict_txt = f.read()
        json_dict = json.loads(json_dict_txt)

    list_poses = json_dict['listPoses']
    person_guid = json_dict['personGuid']

    index = 0
    print('Loading person guid: {0}'.format(person_guid))
    cv2.namedWindow('main_window')
    cv2.namedWindow('main_window_2')

    # Draw global position
    min_x = -200
    max_x = 1000

    min_y = -900
    max_y = 900

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    width_plane = 1000
    height_plane = 500

    img_plane = np.zeros((height_plane, width_plane, 3), np.uint8)

    # Blank image
    img_plane[:, :] = (255, 255, 255)

    while True:
        pose = list_poses[index]

        transformed_points = pose['transformedPoints']
        global_position = pose['globalPosition']
        pose_guid = pose['poseGuid']
        key_pose = pose['keyPose']
        probability = pose['probability']

        print('Index: {0} - poseGuid: {1}'.format(index, pose_guid))
        print('Key Pose: {0} - Probability: {1}'.format(key_pose, probability))

        # View next 4 position distance
        for i in range(index + 1, index + 5):
            if i == len(list_poses):
                break

            point_pos = list_poses[i]['globalPosition']
            dist = ClassUtils.get_euclidean_distance_pt(global_position, point_pos)
            print('Distance: {0}'.format(dist))

        re_scale_factor = 50
        pose_re_scaled = ClassDescriptors.re_scale_pose_factor(transformed_points, re_scale_factor, min_score)
        image_pose = ClassDescriptors.draw_pose_image(pose_re_scaled, min_score, is_transformed=True, key_pose=key_pose)

        cv2.imshow('main_window', image_pose)

        # Draw position
        width_rect = 5

        # Height plane must be set into list
        pt_plane_x = int((global_position[0] - min_x) * width_plane / delta_x)
        pt_plane_y = height_plane - int((global_position[1] - min_y) * height_plane / delta_y)

        pt1 = (pt_plane_x - width_rect, pt_plane_y - width_rect)
        pt2 = (pt_plane_x + width_rect, pt_plane_y + width_rect)

        # Fill rectangle
        cv2.rectangle(img_plane, pt1, pt2, (255, 0, 255), thickness=-1)
        cv2.imshow('main_window_2', img_plane)

        key = cv2.waitKey(0)

        # Transform rectangle to color
        cv2.rectangle(img_plane, pt1, pt2, (0, 0, 0), thickness=-1)

        if key == 27:
            # Esc key
            # Break sequence
            break
        if key == 52:
            # Left arrow
            index -= 1
            if index < 0:
                index = 0
        elif key == 54:
            # Right arrow
            index += 1
            if index >= len(list_poses):
                index = len(list_poses) - 1

    cv2.destroyAllWindows()
    print('Done!')


if __name__ == '__main__':
    main()
