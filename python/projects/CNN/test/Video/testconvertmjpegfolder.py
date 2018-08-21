"""
Script to convert videos
"""
from tkinter import filedialog
import glob
import os
from classmjpegconverter import ClassMjpegConverter
from classutils import ClassUtils


def main():
    print('Initializing main function')
    selection = input('Select 1 to convert remaining videos from mjpeg to mjpegx, Select 2 to reprocess all videos: ')

    if selection == '1':
        reprocess = False
        process_videos(reprocess)
    elif selection == '2':
        reprocess = True
        process_videos(reprocess)
    else:
        print('Selection not recognized: {0}'.format(selection))


def process_videos(reprocess):
    print('Initializing main function')
    print('Warning! This routine will overwrite the selected files')

    init_dir = '/home/mauricio/Videos/Oviedo/'
    options = {'initialdir': init_dir}

    folder = filedialog.askdirectory(**options)

    if folder is None:
        print('Folder not selected')
    else:
        print(folder)
        print('Extracting all mjpeg files')

        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                extension = ClassUtils.get_filename_extension(full_path)

                if extension == '.mjpeg':
                    if not reprocess:
                        mjpegx_path = full_path.replace(".mjpeg", ".mjpegx")
                        if os.path.exists(mjpegx_path):
                            print('Ignoring already converted file {0}'.format(mjpegx_path))
                            continue

                    print('Converting ' + full_path)
                    ClassMjpegConverter.convert_video_mjpeg(full_path)

        print('Done!')


if __name__ == '__main__':
    main()
