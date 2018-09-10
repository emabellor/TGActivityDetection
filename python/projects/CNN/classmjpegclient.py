"""
Mjpeg Client
Inspired in
https://stackoverflow.com/questions/21702477/how-to-parse-mjpeg-http-stream-from-ip-camera

Based in this post to perform cam authentication
https://stackoverflow.com/questions/29708708/http-basic-authentication-not-working-in-python-3-4
"""

from urllib.request import urlopen, Request
import numpy as np
from threading import Thread
from datetime import datetime
import traceback
import base64


class ClassMjpegClient:
    def __init__(self, _url):
        self.url = _url
        self.new_frame_cb = None
        self.new_error_cb = None
        self.id_cam = 0
        self.last_image = np.zeros((0, 0))
        self.last_date = datetime(2000, 1, 1, 0, 0, 0)
        self.first_frame = True
        self.last_movement_frame = datetime(2000, 1, 1, 0, 0, 0)
        self.movement_flag = False
        self.is_playing = False
        self.video_saver_ref = None
        self.username = ''
        self.password = ''

    def set_authentication(self, username, password):
        self.username = username
        self.password = password

    def set_video_saver(self, video_saver_instance):
        self.video_saver_ref = video_saver_instance

    def set_id_cam(self, _id_cam):
        self.id_cam = _id_cam

    def set_last_image_and_date(self, _last_image, _last_date):
        self.last_image = _last_image
        self.last_date = _last_date

    def set_new_frame_cb(self, cb):
        self.new_frame_cb = cb

    def set_new_error_cb(self, cb):
        self.new_error_cb = cb

    def init_stream(self):
        print('Initializing stream')
        Thread(target=self._do_stream).start()

    def _do_stream(self):
        try:
            req = Request(self.url)

            if self.username != '' or self.password != '':
                credentials = ('%s:%s' % (self.username, self.password))
                encoded_credentials = base64.b64encode(credentials.encode('ascii'))
                req.add_header('Authorization', 'Basic %s' % encoded_credentials.decode("ascii"))

            with urlopen(req) as stream:
                byte_arr = bytes()
                self.is_playing = True
                while self.is_playing:
                    byte_arr += stream.read(1024)
                    start = b'\xff\xd8'
                    end = b'\xff\xd9'
                    a = byte_arr.find(start)
                    b = byte_arr.find(end)
                    if a != -1 and b != -1:
                        jpg = byte_arr[a:b + 2]
                        byte_arr = byte_arr[b + 2:]
                        jpg_np = np.fromstring(jpg, dtype=np.uint8)

                        # Calling new frame callback
                        self.new_frame_cb(self, jpg_np)

        except Exception as e:
            print('Exception in url {0} - {1}'.format(self.url, e))
            self.new_error_cb(self, e)

    def close_stream(self):
        self.is_playing = False
