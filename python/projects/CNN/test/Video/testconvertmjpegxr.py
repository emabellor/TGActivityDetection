"""
Video visualizer
Code inspired in stackoverflow response
https://stackoverflow.com/questions/17987598/how-can-i-use-imshow-to-display-multiple-images-in-multiple-windows
"""
from classpeoplereid import ClassPeopleReId
import cv2
import numpy as np
from datetime import datetime
from datetime import timedelta
from classutils import ClassUtils
from classmjpegdate import ClassMjpegDate
from classmjpegconverter import ClassMjpegConverter
from classdescriptors import ClassDescriptors
from tkinter import filedialog
from tkinter import Tk
import tkinter as tk
import shutil
import os
import json
from typing import List


game_period_ms = 500
min_score = 0.05
last_upper = [0, 0, 0]
last_lower = [0, 0, 0]
list_people = list()
list_candidates = list()
count_person = 0
resize_factor = 1.5
image_width = 0
image_height = 0

# Default camera values
list_cams = [419, 420, 421, 428, 429, 430]
date_init = datetime(2018, 2, 24, 14, 15, 0)
date_end = datetime(2018, 2, 24, 15, 15, 0)

CNN_BASE_FOLDER = '/home/mauricio/CNN/Images'
SAVE_IMG_BASE_FOLDER = '/home/mauricio/Pictures/BTF/Examples'
CANDIDATES_FOLDER = '/home/mauricio/Pictures/BTF/Candidates'

base_path = '/home/mauricio/Videos/Oviedo'

# Loading dummy frame
with open('/home/mauricio/Pictures/novideo.jpg', 'rb') as file:
    dummy_frame = file.read()


def main():
    global list_cams
    global date_init
    global date_end
    print('Initializing main function')

    option = input('Selecting list cam option: ')
    if option == '1':
        list_cams = [419, 420, 421, 428, 429, 430]
        date_init = datetime(2018, 2, 24, 14, 15, 0)
        date_end = datetime(2018, 2, 24, 15, 15, 0)
        select_options()
    elif option == '2':
        list_cams = [419, 420, 421, 428, 429, 430]
        date_init = datetime(2018, 7, 27, 15, 0, 0)
        date_end = datetime(2018, 7, 27, 15, 29, 59)
        select_options()
    elif option == '3':
        list_cams = [419, 420]
        date_init = datetime(2018, 9, 4, 3, 0, 0)
        date_end = datetime(2018, 9, 4, 3, 15, 0)
        select_options()
    elif option == '4':
        list_cams = [419, 420, 421]
        date_init = datetime(2018, 9, 3, 12, 45, 0)
        date_end = datetime(2018, 9, 3, 13, 0, 0)
        select_options()
    else:
        print('Option not recognized')


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
                print('Cam calib invalid for id cam {0}'.format(id_cam))
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
    selection = input('Press 1 to generate files, 2 to check files, 3 to perform debugging, 4 to get poses: ')

    if selection == '1':
        generate_files()
    elif selection == '2':
        check_files()
    elif selection == '3':
        debug_reid()
    elif selection == '4':
        get_poses()
    else:
        print('Selection not identified')


def generate_files():
    print('Initializing generate files')

    date_video = date_init
    list_readers = list()
    list_paths = list()
    list_writers = list()

    for i in range(len(list_cams)):
        list_readers.append(ClassMjpegDate(list_cams[i]))
        list_paths.append('')
        list_writers.append(None)

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
                param['poseGuid'] = ClassUtils.generate_uuid()
                param['ticks'] = ClassUtils.datetime_to_ticks(date_video)

            frame_info[2]['listPeople'] = list()
            frame_info_list.append(frame_info)

        # Perform re-identification module
        process_reid(frame_info_list, date_video)

        # Write in image
        for i in range(len(list_cams)):
            date_file = ClassMjpegDate.get_date_file(date_video)
            video_path = ClassMjpegDate.load_path_by_date(date_file, str(list_cams[i]), '.mjpegxr')

            if list_paths[i] != video_path:
                # Open new element in list
                if list_paths[i] != '':
                    list_writers[i].close()

                list_writers[i] = open(video_path, 'wb')
                list_paths[i] = video_path

            frame = frame_info_list[i][0]
            ticks = frame_info_list[i][1]
            json_dict = frame_info_list[i][2]

            ClassMjpegConverter.write_in_file(list_writers[i], ticks, frame, json_dict)

        # Next
        date_video = date_video + timedelta(milliseconds=game_period_ms)

    # Close checking streams
    for writer in list_writers:
        if writer is not None:
            writer.close()

    print('Done!')


def check_files():
    global list_people
    print('Initializing check files')

    date_video = date_init
    list_readers = []
    for i in range(len(list_cams)):
        list_readers.append(ClassMjpegDate(list_cams[i], extension='.mjpegxr'))

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)

    is_playing = True
    forward_until_person = False
    play_factor = 1
    while date_video < date_end:
        frame_info_list = list()

        if is_playing:
            print(date_video)

            for i in range(len(list_cams)):
                frame_info = list_readers[i].load_frame(date_video)
                frame_info_list.append(frame_info)

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
        date_video = process_date_video(key, date_video, is_playing)
        play_factor = process_play_factor(key, play_factor)
        is_playing = process_is_playing(key, is_playing)
        process_save_image(key, date_video, is_playing)
        forward_until_person = process_forward(key, forward_until_person)

    cv2.destroyAllWindows()
    print('Done!')


def process_forward(key, forward_until_person: bool):
    if key == 102:
        # Forward until key
        if forward_until_person:
            print('Disabling forward_until_person')
            return False
        else:
            print('Enabling forward_until_person')
            return True
    else:
        return forward_until_person


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


def process_date_video(key, date_video, is_playing):
    new_date_video = date_video
    if is_playing:
        if key == 52:
            # Left arrow
            # Seconds
            new_date_video = date_video - timedelta(milliseconds=1000 * 3)
            if new_date_video < date_init:
                new_date_video = date_init
        elif key == 54:
            # Right arrow
            # Seconds
            new_date_video = date_video + timedelta(milliseconds=1000 * 3)
        elif key == 49:
            # Arrow 1
            # Minutes
            new_date_video = date_video - timedelta(milliseconds=1000 * 60)
            if new_date_video < date_init:
                new_date_video = date_init
        elif key == 51:
            # Arrow 3
            # Minutes
            new_date_video = date_video + timedelta(milliseconds=1000 * 60)
        elif key == 55:
            # Arrow 7
            # Middle
            new_date_video = date_video - timedelta(milliseconds=1000 * 20)
            if new_date_video < date_init:
                new_date_video = date_init
        elif key == 57:
            new_date_video = date_video + timedelta(milliseconds=1000 * 20)
        else:
            new_date_video = date_video + timedelta(milliseconds=game_period_ms)

    return new_date_video


def process_save_image(key, date_video: datetime, is_playing):
    global list_candidates
    global label
    global ok_params

    if is_playing:
        # Ignore!
        return

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
            global label
            global ok_params

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
                f.write(json.dumps(param))

            # Saving candidate if not exists - Referencing purposes
            if not os.path.exists(CANDIDATES_FOLDER):
                os.makedirs(CANDIDATES_FOLDER)

            file_candidate = os.path.join(CANDIDATES_FOLDER, '{0}.jpg'.format(label))
            if not os.path.exists(file_candidate):
                cv2.imwrite(file_candidate, img_cv)
                candidate_path_json = file_candidate.replace('.jpg', '.json')
                with open(candidate_path_json, 'w') as f:
                    f.write(json.dumps(param))

    # Done writing elements


def process_is_playing(key, is_playing):
    new_is_playing = is_playing
    if key == 53:
        # Don't change date video, but avoid processing
        if is_playing:
            new_is_playing = False
        else:
            new_is_playing = True
    return new_is_playing


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
        if is_playing:
            print(date_video)

            # Getting list frames first
            for i in range(len(list_cams)):
                frame_info = list_readers[i].load_frame(date_video)

                # Generate a guid for every pose
                # Generating blank person guids
                # Reid purposes
                for param in frame_info[2]['params']:
                    param['poseGuid'] = ClassUtils.generate_uuid()
                    param['ticks'] = ClassUtils.datetime_to_ticks(date_video)

                frame_info[2]['listPeople'] = list()
                frame_info_list.append(frame_info)

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

                # Draw poses and vectors4
                draw_images(frame_info_list)

        # Fast selection to forward
        if not forward_until_person:
            key = cv2.waitKey(game_period_ms)
        else:
            key = cv2.waitKey(1)

        if key != -1:
            print('KeyPressed: {0}'.format(key))

        if key == 27:
            # Esc
            break

        # Process elems
        date_video = process_date_video(key, date_video, is_playing)
        play_factor = process_play_factor(key, play_factor)
        is_playing = process_is_playing(key, is_playing)
        process_save_image(key, date_video, is_playing)
        forward_until_person = process_forward(key, forward_until_person)

    cv2.destroyAllWindows()
    print('Done!')


def process_reid(frame_info_list, date_ref: datetime):
    global list_people
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

                    if return_data['distance'] <= 120 and return_data['diffUpper'] < 40 and return_data['diffLower'] < 40:
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
                    total_diff = return_data['diffUpper'] + return_data['diffLower']

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
        list_can: List[ClassPeopleReId] = list()
        for person_last in list_people:
            if person_last not in updated_people:
                score = ClassPeopleReId.compare_people(person, person_last)
                print('Score: {0} - {1} - {2} Color: {3}'.format(person.global_pos, person_last.global_pos, score
                                                                 , person_last.get_rgb_color_str_int()))

                upper, lower, dis = ClassPeopleReId.compare_people_items(person, person_last)
                print('Upper: {0:.4f}, Lower: {1:.4f}, Distance: {2:.4f}'.format(upper, lower, dis))

                if score <= 0.5:
                    list_can.append(person_last)

        # Evaluate candidates for selected elements
        if len(list_can) > 0:
            minimum_score = -1
            selected_person = None

            for person_candidate in list_can:
                score = ClassPeopleReId.compare_people(person, person_candidate)
                if minimum_score == -1 or score < minimum_score:
                    minimum_score = score
                    selected_person = person_candidate

            print('updating {0} with {1}'.format(person.global_pos, selected_person.global_pos))
            selected_person.update_values_from_person(person, date_ref)
            updated_people.append(selected_person)
        else:
            print('Creating new person {0}'.format(person.global_pos))
            list_people.append(person)
            updated_people.append(person)

    # Remove old elements from list
    # No more than 5 seconds
    remove_people: List[ClassPeopleReId] = list()
    for person in list_people:
        delta = date_ref - person.last_date

        if delta.seconds > 5:
            remove_people.append(person)

    for person in remove_people:
        list_people.remove(person)

    # Updating person guids in each frame!
    for reid_person in list_people:
        for frame_info in frame_info_list:
            list_person_guids = frame_info[2]['listPeople']
            if frame_info[2]['camNumber'] == reid_person.cam_number:
                list_person_guids.append(reid_person.get_guid_relation())
                continue

    # Done


def draw_images(frame_info_list):
    global list_candidates
    global count_person

    list_candidates.clear()
    count_person = 0

    # Creating images and drawing vectors
    list_images = list()
    for frame_info in frame_info_list:
        dict_frame = frame_info[2]
        cam_number = dict_frame['camNumber']

        if frame_info is None:
            image_arr = np.frombuffer(dummy_frame, dtype="int32")
            image_cv = cv2.imdecode(image_arr, cv2.IMREAD_ANYCOLOR)
        else:
            image_arr = np.frombuffer(frame_info[0], dtype="int32")
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
    global count_person
    global list_candidates

    # Drawing people into image
    dict_frame = frame_info[2]
    cam_number = dict_frame['camNumber']
    params = dict_frame['params']

    # Draw all poses
    for i, param in enumerate(params):
        vectors = param['vectors']
        local_pos = param['localPosition']

        if ClassUtils.check_vector_integrity_pos(vectors, min_score):
            # Draw valid poses in red and put number
            pt1, pt2 = ClassUtils.get_rectangle_bounds(vectors, min_score)
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

            # Draw image local position
            ClassUtils.draw_position(image, local_pos)

        ClassUtils.draw_pose(image, vectors, min_score)

    # Done drawing vectors


def draw_people(list_images):
    global list_people

    for person in list_people:
        image_cv = None
        cam_number = person.cam_number

        for image in list_images:
            if cam_number == image['camNumber']:
                image_cv = image['imageCv']
                break

        if image_cv is None:
            raise Exception('Cant find image for camNumber: {0}'.format(cam_number))

        global last_upper
        global last_lower

        # Draw elements into image
        pt1, pt2 = ClassUtils.get_rectangle_bounds(person.vectors, min_score)
        # cv2.rectangle(image_cv, pt1, pt2, person.get_bgr_color(), 5)

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

        last_upper = person.color_upper
        last_lower = person.color_lower


def show_images(list_images):
    global resize_factor
    global image_width
    global image_height

    if len(list_images) > 6:
        raise Exception('List images must be less or equal than 6')

    base_iter = 3
    image_up = None
    image_down = None

    for i in range(base_iter):
        if image_up is None:
            image_up = list_images[i]['imageCv']
        else:
            image_up = np.hstack((image_up, list_images[i]['imageCv']))

    for i in range(base_iter, len(list_images)):
        if image_down is None:
            image_down = list_images[i]['imageCv']
        else:
            image_down = np.hstack((image_down, list_images[i]['imageCv']))

    result_image = np.vstack((image_up, image_down))

    new_y = int(result_image.shape[0] / resize_factor)
    new_x = int(result_image.shape[1] / resize_factor)

    image_height = int(new_y / 1)
    image_width = int(new_x / 3)

    result_image = cv2.resize(result_image, (new_x, new_y))
    cv2.imshow('main_window', result_image)


def get_poses():
    print('Generating poses from videos')
    print('WARNING: Cleaning base folder {0}'.format(CNN_BASE_FOLDER))

    # Cleaning base tree folder
    if os.path.exists(CNN_BASE_FOLDER):
        shutil.rmtree(CNN_BASE_FOLDER)
    os.makedirs(CNN_BASE_FOLDER)

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
        image_folder = os.path.join(CNN_BASE_FOLDER, person.get_person_guid())
        if not os.path.isdir(image_folder):
            os.makedirs(image_folder)

        # List poses
        file_path = os.path.join(image_folder, 'poses.json')

        list_poses = list()
        for person_item in list_person:
            list_poses.append(person_item.person_param)

        print('Writing file: {0}'.format(file_path))
        with open(file_path, 'w') as f:
            f.write(json.dumps(list_poses))

    print('Done!')


if __name__ == '__main__':
    main()
