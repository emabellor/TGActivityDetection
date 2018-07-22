"""
File must have all elements
"""
from classmjpegconverter import ClassMjpegConverter


def main():
    print('Processing video from list')

    input_video = '/home/mauricio/Videos/mjpeg/11-00-00.mjpeg'

    print('Converting mjpeg from input_video ' + input_video)
    ClassMjpegConverter.convert_video_mjpeg(input_video)

    print('Done!')


if __name__ == '__main__':
    main()
