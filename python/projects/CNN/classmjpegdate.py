"""
ClassMjpegDate
Frame selector
"""
from datetime import datetime
from os import path
from classmjpegreader import ClassMjpegReader
from classmjpegconverter import ClassMjpegConverter
from classutils import ClassUtils
import logging

video_base_path = '/home/mauricio/Videos/Oviedo'
logger = logging.getLogger('ClassMjpegDate')


class ClassMjpegDate:

    def __init__(self, cam_number):
        logger.info('Initializing class MjpegDate')
        self._cam_number = cam_number
        self._frame_info_list = []
        self._frame_info = []
        self._frame_date = datetime.now()
        self._last_frame = None

    def load_frame(self, date: datetime):
        logger.debug(self._frame_date)

        logger.debug('Get frame to load')

        minutes = int(date.minute / 15) * 15
        seconds = 0
        microseconds = 0
        date_file = date.replace(minute=minutes, second=seconds, microsecond=microseconds)

        # Check fist if date is in list
        found = False

        for frame_item in self._frame_info_list:
            date_video = frame_item['date_video']
            if date_video == date_file:
                found = True
                self._frame_date = date_file
                self._frame_info = frame_item['frame_info']
                break

        if not found:
            logger.debug('Loading file name')
            logger.debug(date_file)
            file_path = self._load_path_by_date(date_file)

            if not path.exists(file_path):
                raise Exception('Path does not exists {0}'.format(file_path))
            else:
                logger.debug('Loading video from path')
                frame_info = ClassMjpegReader.process_video_mjpegx(file_path)

                self._frame_info = frame_info
                self._frame_date = date_file

                logger.debug('Frame counter: {0}'.format(len(frame_info)))

                # Must be true -> First frame to send -> Avoid exceptions
                if self._last_frame is None:
                    self._last_frame = frame_info[0]

                self._frame_info_list.append({
                    'date_video': date_file,
                    'frame_info': frame_info
                })

        return self._load_frame_by_date(date)

    def update_frame(self, frame_to_update):
        # updating frame by ticks
        found = False
        for frame_item in self._frame_info_list:
            frame_info = frame_item['frame_info']

            for frame_item_list in frame_info:
                # Compare Ticks
                if frame_item_list[1] == frame_to_update[1]:
                    found = True
                    frame_item_list = frame_to_update
                    break

        if not found:
            raise Exception('Cant find frame to update with ticks {0}'.format(frame_to_update[1]))

    def save_frame_info(self):
        for frame_item in self._frame_info_list:
            frame_info = frame_item['frame_info']
            date_video = frame_item['date_video']

            path_video = self._load_path_by_date(date_video)

            # Converting video from list
            ClassMjpegConverter.save_video_from_list_frames(path_video, frame_info)
            logger.info('Video saved in path {0}'.format(path_video))

        logger.debug('Done converting videos')

    def _load_path_by_date(self, date: datetime):
        logger.debug('Generating path by date')
        file_name = date.strftime('%H-%M-%S') + '.mjpegx'
        file_path = path.join(video_base_path, date.strftime('%Y-%m-%d'), str(self._cam_number), file_name)
        logger.debug(file_path)
        return file_path

    def _load_frame_by_date(self, date: datetime):
        """
        Assume that frames are organized
        """
        min_delta = -1
        logger.debug('Loading frame by date')

        selected_frame = None
        for frame in self._frame_info:
            ticks = frame[1]

            date_frame = ClassUtils.ticks_to_datetime(ticks)
            delta = date - date_frame

            if min_delta == -1 or delta < min_delta:
                if delta.total_seconds() > 0:
                    min_delta = delta
                    selected_frame = frame

        if selected_frame is None:
            selected_frame = self._last_frame

        self._last_frame = selected_frame
        return selected_frame
