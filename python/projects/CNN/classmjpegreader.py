import json
import os
import logging

logger = logging.getLogger('ClassMjpegReader')


class ClassMjpegReader:

    @staticmethod
    def process_video(video_path):
        extension = os.path.splitext(video_path)[1]

        if extension != '.mjpeg':
            raise Exception('Video must have .mjpeg extension')
        else:
            logger.debug('Opening video')
            list_images = []

            with open(video_path, mode='rb') as file:  # b is important -> binary
                file_content = file.read()
                offset = 0

                while offset < len(file_content):
                    # Reading first element
                    file_size_bin = file_content[offset:offset+4]
                    file_size = int.from_bytes(file_size_bin, byteorder='little', signed=False)
                    offset += 4

                    # Reading ticks -> Not counting for now
                    ticks_bin = file_content[offset:offset+8]
                    ticks = int.from_bytes(ticks_bin, byteorder='little')
                    offset += 8

                    # Reading image
                    image = file_content[offset:offset + file_size]
                    offset += file_size

                    list_images.append((image, ticks))

            return list_images

    @staticmethod
    def process_video_mjpegx(video_path):
        extension = os.path.splitext(video_path)[1]

        if extension != '.mjpegx' and extension != '.mjpegxr':
            raise Exception('Video must have .mjpegx extension or .mjpegxr extension')
        else:
            logger.info('Opening video {0}'.format(video_path))
            list_images = []

            with open(video_path, mode='rb') as file:
                file_content = file.read()
                offset = 0

                while offset < len(file_content):
                    # Reading first element
                    file_size_bin = file_content[offset:offset + 4]
                    file_size = int.from_bytes(file_size_bin, byteorder='little', signed=False)
                    offset += 4

                    # Reading ticks -> Not counting for now
                    ticks_bin = file_content[offset:offset + 8]
                    ticks = int.from_bytes(ticks_bin, byteorder='little')
                    offset += 8

                    # Reading image + dict object
                    bin_array = file_content[offset:offset + file_size]
                    offset += file_size

                    # Reading dict object size
                    dict_len_bin = bin_array[len(bin_array) - 4: len(bin_array)]
                    dict_len = int.from_bytes(dict_len_bin, byteorder='little')

                    logger.debug('File size: ' + str(file_size))
                    logger.debug('Dict size: ' + str(dict_len))

                    # Reading dict array and image
                    image = bin_array[0: file_size - dict_len - 4]
                    dict_bin = bin_array[file_size - dict_len - 4: file_size - 4]

                    dict_json = dict_bin.decode()
                    logger.debug(dict_json)

                    list_images.append((image, ticks, json.loads(dict_json)))

            return list_images

