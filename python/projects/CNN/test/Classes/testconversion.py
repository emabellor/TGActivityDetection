import time
from classutils import ClassUtils
from classmjpegconverter import ClassMjpegConverter
import os


def main():
    print('Initializing conversion function')
    folder = ClassUtils.video_base_path

    while True:
        print('Checking videos')
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                extension = ClassUtils.get_filename_extension(full_path)

                if extension == '.mjpeg':
                    print(file)
                    print('Reprocessing ' + full_path + ' to mjpegx')
                    camera_number = ClassUtils.get_cam_number_from_path(full_path)
                    ClassMjpegConverter.convert_video_mjpeg(full_path)

        print('Done checking - Waiting for 5 secs')
        time.sleep(5)


if __name__ == '__main__':
    main()
