from classmjpegclient import ClassMjpegClient
import numpy as np
from threading import Timer
import cv2
from datetime import datetime
from classvideosaver import ClassVideoSaver
from classmjpegconverter import ClassMjpegConverter
from classutils import ClassUtils
import logging
import time
import os

logger = logging.getLogger('Main')

period_milliseconds = 500
width_resize = 320
height_resize = 240
counter = 0
close = False

list_cams = [
    {
        'idCam': 597,
        'urlCam': '',
        'username': '',
        'password': ''
    },
    {
        'idCam': 598,
        'urlCam': '',
        'username': '',
        'password': ''
    },
    {
        'idCam': 599,
        'urlCam': '',
        'username': '',
        'password': ''
    },
    {
        'idCam': 605,
        'urlCam': '',
        'username': '',
        'password': ''
    },
    {
        'idCam': 606,
        'urlCam': '',
        'username': '',
        'password': ''
    },
    {
        'idCam': 607,
        'urlCam': '',
        'username': '',
        'password': ''
    },
    {
        'idCam': 1000,
        'urlCam': '',
        'username': '',
        'password': ''
    }
]


def new_frame(obj: ClassMjpegClient, frame: np.ndarray):
    # Evaluating timestamp
    global counter
    now = datetime.now()
    delta_milli = (now - obj.last_date).total_seconds()*1000

    if delta_milli >= period_milliseconds:
        # Done - Calculating new timestamp
        milli_sec = now.microsecond / 1000
        res = milli_sec % period_milliseconds
        milli_sec -= res

        # Set datetime
        new_date = now.replace(microsecond=int(milli_sec*1000))

        # Decoding and process
        image_cv = cv2.imdecode(frame, cv2.IMREAD_GRAYSCALE)
        image_res = cv2.resize(image_cv, (width_resize, height_resize))

        if obj.last_image.shape != (0, 0):
            diff = cv2.absdiff(image_res, obj.last_image)
            _, threshold = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

            sum_val = np.sum(threshold)
            movement = (sum_val * 100) / (255 * width_resize * height_resize)
        else:
            # Avoid first image problem
            movement = 0

        obj.set_last_image_and_date(image_res, new_date)

        # Generating movement analysis
        mov_threshold = 5
        save_frame = False
        seconds_after_mov = 5
        key_frame_period_sec = 30

        if movement >= mov_threshold or obj.first_frame:
            obj.first_frame = False

            obj.movement_flag = True
            obj.last_movement_frame = now
            save_frame = True
        else:
            seconds_elapsed = (now - obj.last_movement_frame).total_seconds()
            if obj.movement_flag:
                save_frame = True

                if seconds_elapsed >= seconds_after_mov:
                    obj.movement_flag = False
                    obj.last_movement_frame = now
            else:
                if seconds_elapsed >= key_frame_period_sec:
                    obj.last_movement_frame = now
                    save_frame = True

        if save_frame:
            print('Saving frame for camera {0} {1}'.format(obj.id_cam, now))

            # Saving frame for camera into list
            obj.video_saver_ref.add_frame(frame, now)
            counter += 1


def init_stream(obj: ClassMjpegClient):
    if not close:
        print('Initializing again stream: {0}'.format(obj.url))
        obj.init_stream()


def new_error(obj: ClassMjpegClient, exception):
    global close
    print('New error - Waiting for 5 seconds to reopen')

    sec_init_again = 5
    Timer(sec_init_again, lambda: init_stream(obj)).start()


def main():
    FORMAT = "%(asctime)s [%(name)-16.16s] [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(level=logging.ERROR, format=FORMAT)

    global counter
    global close
    global list_cams
    list_instances_grabber = list()
    date_init = datetime.now()

    logger.debug('Initializing main function')

    # Generating url cam into list
    for item_cam in list_cams:
        id_cam = item_cam['idCam']
        url_cam = item_cam['urlCam']

        if url_cam == '':
            prefix = int(id_cam / 100)
            suffix = id_cam % 100

            url_cam = 'http://192.168.1{0}.1{1}/videostream.cgi?rate=6&user=admin&pwd='.format(
                str(prefix).zfill(2), str(suffix).zfill(2)
            )
            item_cam['urlCam'] = url_cam

    # Checking period integrity
    if 1000 % period_milliseconds != 0:
        raise Exception('Invalid period_milliseconds value')

    for cam in list_cams:
        id_cam = cam['idCam']
        url_cam = cam['urlCam']
        username = cam['username']
        password = cam['password']

        instance_grabber = ClassMjpegClient(url_cam)
        instance_grabber.set_id_cam(id_cam)
        instance_grabber.set_authentication(username, password)

        # Initializing instance save
        instance_saver = ClassVideoSaver(id_cam)
        instance_grabber.set_video_saver(instance_saver)

        instance_grabber.set_new_frame_cb(new_frame)
        instance_grabber.set_new_error_cb(new_error)

        instance_grabber.init_stream()
        list_instances_grabber.append(instance_grabber)

    # Generate video conversion thread

    input('Done! Press enter to stop recording')
    close = True
    date_end = datetime.now()

    # When key pressed, stop all and write videos
    for instance in list_instances_grabber:
        instance.close_stream()
        instance.video_saver_ref.save_data()

    print('Done!')
    print('Saved frames: {0}'.format(counter))
    print('Elapsed: {0}'.format((date_end - date_init).total_seconds()))


if __name__ == '__main__':
    main()

