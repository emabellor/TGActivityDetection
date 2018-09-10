"""
Video visualizer
Code inspired in stackoverflow response
https://stackoverflow.com/questions/17987598/how-can-i-use-imshow-to-display-multiple-images-in-multiple-windows
"""
from classmjpegdate import ClassMjpegDate
from classpeoplereid import ClassPeopleReId
import cv2
import numpy as np
from datetime import datetime
from datetime import timedelta
from classutils import ClassUtils

game_period_ms = 500
min_score = 0.05
last_upper = [0, 0, 0]
last_lower = [0, 0, 0]
list_people = list()
play_factor = 1

# Loading dummy frame
with open('/home/mauricio/Pictures/novideo.jpg', 'rb') as file:
    dummy_frame = file.read()


def main():
    global play_factor
    print('Warning - This code is deprecated')
    print('Use testconvertmjpegxr instead')

    print('Initializing main function')

    list_cams = [419, 420, 421, 428, 429, 430]
    date_init = datetime(2018, 2, 24, 14, 15, 0)
    date_end = datetime(2018, 2, 24, 15, 15, 0)

    date_video = date_init
    list_readers = []
    for i in range(len(list_cams)):
        list_readers.append(ClassMjpegDate(list_cams[i]))

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)

    is_playing = True
    while date_video < date_end:
        if is_playing:
            print(date_video)

            # Getting list frames first
            frame_info_list = list()
            for i in range(len(list_cams)):
                frame_info = list_readers[i].load_frame(date_video)
                frame_info_list.append(frame_info)

            # Perform re-identification module
            process_reid(frame_info_list, date_video)

            # Draw poses and vectors4
            draw_images(frame_info_list)

        # Waiting for pressed key
        key = cv2.waitKey(game_period_ms)

        if key != -1:
            print('KeyPressed: {0}'.format(key))

        if key == 27:
            # Esc
            break
        elif key == 52:
            # Left arrow
            # Seconds
            date_video = date_video - timedelta(milliseconds=1000 * 3)
            if date_video < date_init:
                date_video = date_init
        elif key == 54:
            # Right arrow
            # Seconds
            date_video = date_video + timedelta(milliseconds=1000 * 3)
        elif key == 49:
            # Arrow 1
            # Minutes
            date_video = date_video - timedelta(milliseconds=1000 * 60)
            if date_video < date_init:
                date_video = date_init
        elif key == 51:
            # Arrow 3
            # Minutes
            date_video = date_video + timedelta(milliseconds=1000 * 60)
        elif key == 55:
            # Arrow 7
            # Middle
            date_video = date_video - timedelta(milliseconds=1000 * 20)
            if date_video < date_init:
                date_video = date_init
        elif key == 57:
            date_video = date_video + timedelta(milliseconds=1000 * 20)
        elif key == 53:
            # Don't change date video, but avoid processing
            if is_playing:
                is_playing = False
            else:
                is_playing = True
        elif key == 190:
            # Change playing factor
            play_factor = 1
            print('Set play_factor to {0}'.format(play_factor))
        elif key == 191:
            play_factor = 2
            print('Set play_factor to {0}'.format(play_factor))
        elif key == 193:
            play_factor = 4
            print('Set play_factor to {0}'.format(play_factor))
        else:
            if is_playing:
                date_video = date_video + timedelta(milliseconds=game_period_ms)

    cv2.destroyAllWindows()
    print('Done!')


def process_reid(frame_info_list, date_video):
    global list_people
    list_new_people = list()

    # Load list people
    for frame_info in frame_info_list:
        list_new_people += ClassPeopleReId.load_people_from_frame_info(frame_info, date_video)

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
    updated_people = list()

    # Re identification part
    # Find candidates
    # Select less score
    for person in list_new_people:
        list_candidates = list()
        for person_last in list_people:
            if person_last not in updated_people:
                score = ClassPeopleReId.compare_people(person, person_last)
                print('Score: {0} - {1} - {2} Color: {3}'.format(person.global_pos, person_last.global_pos, score
                                                                 , person_last.get_rgb_color_str_int()))

                upper, lower, dis = ClassPeopleReId.compare_people_items(person, person_last)
                print('Upper: {0:.4f}, Lower: {1:.4f}, Distance: {2:.4f}'.format(upper, lower, dis))

                if score <= 0.5:
                    list_candidates.append(person_last)

        # Evaluate candidates for selected elems
        if len(list_candidates) > 0:
            minimum_score = -1
            selected_person = None

            for person_candidate in list_candidates:
                score = ClassPeopleReId.compare_people(person, person_candidate)
                if minimum_score == -1 or score < minimum_score:
                    minimum_score = score
                    selected_person = person_candidate

            print('updating {0} with {1}'.format(person.global_pos, selected_person.global_pos))
            selected_person.update_values_from_person(person)
            updated_people.append(selected_person)
        else:
            print('Creating new person {0}'.format(person.global_pos))
            list_people.append(person)
            updated_people.append(person)

    # Remove old elements from list
    remove_people = list()
    for person in list_people:
        delta = datetime.now() - person.last_date

        if delta.seconds > 5:
            remove_people.append(person)

    for person in remove_people:
        list_people.remove(person)

    # Done


def draw_images(frame_info_list):
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
            draw_vectors(image_cv, frame_info)

        list_images.append({
            'imageCv': image_cv,
            'camNumber': cam_number
        })

    # Drawing person info
    draw_people(list_images)

    # Show list images in window
    show_images(list_images)


def draw_vectors(image: np.ndarray, frame_info):
    global list_people

    # Drawing people into image
    dict_frame = frame_info[2]
    vectors = dict_frame['vectors']

    # Draw all poses
    for vector in vectors:
        ClassUtils.draw_pose(image, vector, min_score)


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

        last_upper = person.color_upper
        last_lower = person.color_lower


def show_images(list_images):
    if len(list_images) % 2 != 0:
        raise Exception('List images must be pair in size')

    if len(list_images) == 0:
        raise Exception('Len image must be greater than zero')

    base_iter = int(len(list_images) / 2)
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
    resize_factor = 1
    new_y = int(result_image.shape[0] / resize_factor)
    new_x = int(result_image.shape[1] / resize_factor)

    result_image = cv2.resize(result_image, (new_x, new_y))
    cv2.imshow('main_window', result_image)


if __name__ == '__main__':
    main()
