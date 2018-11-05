"""
ClassMjpegDate
Frame selector
"""
from datetime import datetime
from os import path
from classmjpegreader import ClassMjpegReader
from classmjpegconverter import ClassMjpegConverter
from classutils import ClassUtils
from sys import platform
import logging
import math

logger = logging.getLogger('ClassMjpegDate')


class ClassMjpegDate:
    def __init__(self, cam_number, extension='.mjpegx'):
        logger.info('Initializing class MjpegDate')
        self._cam_number = cam_number
        self._frame_info_list = []
        self._frame_info = []
        self._frame_date = datetime.now()
        self._last_frame = None
        self._extension = extension
        self._last_index = 0

    def load_frame(self, date: datetime):
        logger.debug(self._frame_date)

        logger.debug('Get frame to load')

        date_file = ClassUtils.get_date_file(date)

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
            file_path = ClassUtils.load_path_by_date(date_file, self._cam_number, self._extension)

            if not path.exists(file_path):
                # Avoid exceptions - Video partial
                print('Path does not exists {0}'.format(file_path))
                frame_info = []
                self._frame_info = frame_info
                self._frame_date = date_file

                if self._last_frame is None:
                    raise Exception('Fist video must exist!')

                self._frame_info_list.append({
                    'date_video': date_file,
                    'frame_info': frame_info
                })
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

            path_video = ClassUtils.load_path_by_date(date_video, self._cam_number, self._extension)

            # Converting video from list
            ClassMjpegConverter.save_video_from_list_frames(path_video, frame_info)
            logger.info('Video saved in path {0}'.format(path_video))

        logger.debug('Done converting videos')

    def _load_frame_by_date(self, date: datetime):
        """
        Assume that frames are organized
        """
        min_delta = -1
        logger.debug('Loading frame by date')
        found = False

        # Improves processing time
        self._last_index += 1
        selected_frame = None

        if self._last_index < len(self._frame_info):
            frame = self._frame_info[self._last_index]
            ticks = frame[1]

            date_frame = ClassUtils.ticks_to_datetime(ticks)
            delta = math.fabs((date - date_frame).seconds)

            if delta < 0.5:
                selected_frame = frame
                found = True

        # Iterate over all elems
        if selected_frame is None:
            for index, frame in enumerate(self._frame_info):
                ticks = frame[1]

                date_frame = ClassUtils.ticks_to_datetime(ticks)
                delta = date - date_frame

                if min_delta == -1 or delta < min_delta:
                    if delta.total_seconds() > 0:
                        min_delta = delta
                        selected_frame = frame
                        self._last_index = index

        # Get last if not found
        if selected_frame is None:
            selected_frame = self._last_frame

        # Add found variable to dir
        selected_frame[2]['found'] = found

        self._last_frame = selected_frame
        return selected_frame
