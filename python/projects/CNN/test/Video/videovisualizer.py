"""
Video visualizer
Code inspired in stackoverflow response
https://stackoverflow.com/questions/17987598/how-can-i-use-imshow-to-display-multiple-images-in-multiple-windows
"""
from classmjpegdate import ClassMjpegDate
import cv2
import numpy as np
from datetime import datetime
from datetime import timedelta
from classutils import ClassUtils

game_period_ms = 500


def main():
    print('Initializing main function')

    list_cams = [419, 420, 421, 430, 429, 428]
    date_init = datetime(2018, 2, 24, 14, 15, 0)
    date_end = datetime(2018, 2, 24, 15, 15, 0)

    date_video = date_init
    list_readers = []
    for i in range(len(list_cams)):
        list_readers.append(ClassMjpegDate(list_cams[i]))

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    with open('/home/mauricio/Pictures/novideo.jpg', 'rb') as file:
        dummy_frame = file.read()

    while date_video < date_end:
        print(date_video)

        list_images = list()
        for i in range(len(list_cams)):
            frame_info = list_readers[i].load_frame(date_video)

            if frame_info is None:
                image = np.frombuffer(dummy_frame, dtype="int32")
                image_cv = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
            else:
                image = np.frombuffer(frame_info[0], dtype="int32")
                image_cv = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
                draw_people(image_cv, frame_info)

            list_images.append(image_cv)

        show_images(list_images)
        key = cv2.waitKey(game_period_ms)
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

        else:
            date_video = date_video + timedelta(milliseconds=game_period_ms)

    cv2.destroyAllWindows()
    print('Done!')


def draw_people(image: np.ndarray, frame_info):
    # Drawing people into image
    dict_frame = frame_info[2]

    frame_position = dict_frame['positions']
    vectors = dict_frame['vectors']

    for candidate_position in frame_position:
        score = candidate_position[2]
        index = frame_position.index(candidate_position)
        person_vectors = vectors[index]
        pos_vector = frame_position[index]

        # Only takes into account scores in one
        if score == 1:
            # Draw elements into image
            min_score = 0.05
            pt1, pt2 = ClassUtils.get_rectangle_bounds(person_vectors, min_score)
            cv2.rectangle(image, pt1, pt2, (0, 0, 255), 5)

            # Draw position vector
            font = cv2.FONT_HERSHEY_SIMPLEX
            pos_txt = (pt2[0], pt1[1])
            font_scale = 0.6
            font_color = (255, 255, 255)
            line_type = 2

            cv2.putText(image, '({0}, {1})'.format(pos_vector[0], pos_vector[1]),
                        pt1,
                        font,
                        font_scale,
                        font_color,
                        line_type)


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
            image_up = list_images[i]
        else:
            image_up = np.hstack((image_up, list_images[i]))

    for i in range(base_iter, len(list_images)):
        if image_down is None:
            image_down = list_images[i]
        else:
            image_down = np.hstack((image_down, list_images[i]))

    result_image = np.vstack((image_up, image_down))
    resize_factor = 1.5
    new_y = int(result_image.shape[0] / resize_factor)
    new_x = int(result_image.shape[1] / resize_factor)

    result_image = cv2.resize(result_image, (new_x, new_y))
    cv2.imshow('main_window', result_image)


if __name__ == '__main__':
    main()
