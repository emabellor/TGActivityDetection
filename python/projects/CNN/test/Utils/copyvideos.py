"""
Script to copy files from servers
"""

from datetime import datetime
from datetime import timedelta
from shutil import copyfile
from sys import platform
import os

if platform == 'linux' or platform == 'linux2':
    base_path = '/home/mauricio/Videos/Oviedo'
    output_path = '/home/mauricio/Videos/Test'
else:
    base_path = 'H:\\VideosCI24\\'
    output_path = 'C:\\DownloadedVideos\\'

list_cams = [419, 420, 421, 428, 429, 430]
date_init = datetime(2018, 9, 29, 12, 30, 0)
date_end = datetime(2018, 9, 29, 12, 59, 59)


def main():
    print('Initializing main function')

    print('Base path: {0}'.format(base_path))
    print('Output path: {0}'.format(output_path))
    print('List cams: {0}'.format(list_cams))
    print('Date init: {0}'.format(date_init))
    print('Date end: {0}'.format(date_end))

    res = input('Is it okay? (y/n): ')

    if res == 'y' or res == 'Y':
        copy_files()
    else:
        print('Exit!')


def copy_files():
    print('Copying files!')
    list_copied_files = list()

    current_date = date_init

    while current_date < date_end:
        date_file = get_date_file(current_date)
        for cam in list_cams:
            cam_str = str(cam)

            base_file = load_path_by_date(base_path, date_file, cam_str, '.mjpeg')
            output_file = load_path_by_date(output_path, date_file, cam_str, '.mjpeg')

            if base_file not in list_copied_files:
                output_dir = os.path.dirname(output_file)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                print('Copying file: {0}'.format(base_file))

                if os.path.exists(base_file):
                    copyfile(base_file, output_file)
                else:
                    print('Ignoring file: {0}'.format(base_file))

                list_copied_files.append(base_file)

        current_date = current_date + timedelta(seconds=60)

    print('Done! {0} files copied'.format(len(list_copied_files)))


# Copy class for compatibility reasons
def get_date_file(date):
    minutes = int(date.minute / 15) * 15
    seconds = 0
    microseconds = 0
    date_file = date.replace(minute=minutes, second=seconds, microsecond=microseconds)
    return date_file


# Copy class for compatibility reasons
def load_path_by_date(video_base_path: str, date: datetime, cam_number: str, extension: str):
    file_name = date.strftime('%H-%M-%S') + extension
    file_path = os.path.join(video_base_path, date.strftime('%Y-%m-%d'), str(cam_number), file_name)
    return file_path


if __name__ == '__main__':
    main()

