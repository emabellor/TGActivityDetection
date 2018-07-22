from tkinter import filedialog
import glob
import os
from classmjpegconverter import ClassMjpegConverter
from tkinter import Tk


def main():
    print('Initializing main function')

    camera_number = input('Insert the camera number: ')

    Tk().withdraw()

    init_dir = '/home/mauricio/Videos/Oviedo/2018-02-24/' + camera_number
    if not os.path.isdir(init_dir):
        init_dir = '/home/mauricio/Videos/Oviedo/'

    options = {'initialdir': init_dir}

    folder = filedialog.askdirectory(**options)

    if folder is None:
        print('Folder not selected')
    else:
        print(folder)
        print('Extracting all mjpegx files')

        os.chdir(folder)
        files = glob.glob("*.mjpegx")

        for file in files:
            full_path = os.path.join(folder, file)
            print(files)
            print('Converting ' + full_path + ' to mjpegx')
            ClassMjpegConverter.convert_video_mjpegx(full_path, camera_number)

        print('Done!')


if __name__ == '__main__':
    main()
