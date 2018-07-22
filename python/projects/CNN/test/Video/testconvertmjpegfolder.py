"""
Script to convert videos
"""
from tkinter import filedialog
import glob
import os
from classmjpegconverter import ClassMjpegConverter


def main():
    print('Initializing main function')
    print('Warning! This routine will overwrite the selected files')

    camera_number = input('Insert the camera number: ')

    init_dir = '/home/mauricio/Videos/Oviedo/2018-02-24/' + camera_number
    if not os.path.isdir(init_dir):
        init_dir = '/home/mauricio/Videos/Oviedo/'

    options = {'initialdir': init_dir}

    folder = filedialog.askdirectory(**options)

    if folder is None:
        print('Folder not selected')
    else:
        print(folder)
        print('Extracting all mjpeg files')

        os.chdir(folder)
        files = glob.glob("*.mjpeg")

        for file in files:
            full_path = os.path.join(folder,file)
            print(files)
            print('Converting ' + full_path)
            ClassMjpegConverter.convert_video_mjpeg(full_path)

        print('Done!')


if __name__ == '__main__':
    main()
