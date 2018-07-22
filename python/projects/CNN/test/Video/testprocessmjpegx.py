from classmjpegconverter import ClassMjpegConverter


def main():
    print('Processing video from list')

    input_video = '/home/mauricio/Videos/mjpeg/11-00-00.mjpegx'
    cam_number = '419'

    print('Converting mjpegx from input_video ' + input_video)
    ClassMjpegConverter.convert_video_mjpegx(input_video, cam_number)

    print('Done!')


if __name__ == '__main__':
    main()

