"""
Files to use custom capture:
capturecamusb.py
capturemjpeg.py
Cam 99 is used for custom capture!
"""

import cv2
from datetime import datetime
import numpy as np
from classvideosaver import ClassVideoSaver

period_milliseconds = 500
width_resize = 320
height_resize = 240
counter = 0
last_image = np.zeros((0, 0))
first_frame = True
movement_flag = False
last_movement_frame = False
current_date = datetime(2000, 1, 1, 0, 0, 0)


def main():
    global current_date

    print('Initialize main function')

    cam_cv_index = input('Select cam cv index: ')
    cam_cv_index = int(cam_cv_index)
    print('Cam cv index: {0}'.format(cam_cv_index))

    cam_number = input('Select cam number: ')
    print('Cam number: {0}'.format(cam_number))

    # Initializing capture instance
    instance_saver = ClassVideoSaver(cam_number)

    # Initializing video instance
    cap = cv2.VideoCapture(cam_cv_index)

    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        now = datetime.now()

        total_ms = (now - current_date).total_seconds() * 1000
        if total_ms > period_milliseconds:
            new_frame(frame, cam_number, instance_saver)

        cv2.imshow('main_window', frame)
        key = cv2.waitKey(10)

        if key != -1:
            print('Key: {0}'.format(key))

            if key == 27:
                break

    # Loading current data into memory
    instance_saver.save_data()
    instance_saver.change_extension()
    cv2.destroyAllWindows()
    print('Done!')


def new_frame(frame: np.ndarray, id_cam: str, video_saver_ref: ClassVideoSaver):
    # Evaluating timestamp
    global counter
    global last_image
    global first_frame
    global movement_flag
    global last_movement_frame
    global current_date

    now = datetime.now()
    delta_milli = (now - current_date).total_seconds()*1000

    # Done - Calculating new timestamp
    milli_sec = now.microsecond / 1000
    res = milli_sec % period_milliseconds
    milli_sec -= res

    # Set datetime
    new_date = now.replace(microsecond=int(milli_sec*1000))

    # Decoding and process
    image_res = cv2.resize(frame, (width_resize, height_resize))

    if last_image.shape != (0, 0):
        diff = cv2.absdiff(image_res, last_image)
        _, threshold = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

        sum_val = np.sum(threshold)
        movement = (sum_val * 100) / (255 * width_resize * height_resize)
    else:
        # Avoid first image problem
        movement = 0

    last_image = image_res
    current_date = new_date

    # Generating movement analysis
    mov_threshold = 5
    save_frame = False
    seconds_after_mov = 5
    key_frame_period_sec = 30

    if movement >= mov_threshold or first_frame:
        first_frame = False
        movement_flag = True
        last_movement_frame = now
        save_frame = True
    else:
        seconds_elapsed = (now - last_movement_frame).total_seconds()
        if movement_flag:
            save_frame = True

            if seconds_elapsed >= seconds_after_mov:
                movement_flag = False
                last_movement_frame = now
        else:
            if seconds_elapsed >= key_frame_period_sec:
                last_movement_frame = now
                save_frame = True

    if save_frame:
        print('Saving frame for camera {0} {1} mov: {2}'.format(id_cam, now, movement))

        # Saving frame for camera into list
        _, buffer = cv2.imencode('.jpg', frame)
        video_saver_ref.add_frame(buffer, now)
        counter += 1


if __name__ == '__main__':
    main()
