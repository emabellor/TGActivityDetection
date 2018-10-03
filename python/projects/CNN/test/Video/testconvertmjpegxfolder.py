from tkinter import filedialog
import glob
import os
from classmjpegconverter import ClassMjpegConverter
from tkinter import Tk
from classutils import ClassUtils
from classnn import ClassNN


def main():
    print('Initializing main function')
    print('Warning - You must convert to mjpegx first')

    # Withdrawing Tkinter window
    Tk().withdraw()

    init_dir = '/home/mauricio/Videos/Oviedo/'
    options = {'initialdir': init_dir}

    folder = filedialog.askdirectory(**options)

    if folder is None:
        print('Folder not selected')
    else:
        print(folder)

        # Initializing pose instance
        instance_nn_pose = ClassNN(model_dir=ClassNN.model_dir_pose,
                                   classes=ClassNN.classes_num_pose,
                                   hidden_number=ClassNN.hidden_num_pose)

        print('Extracting all mjpegx files')

        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                extension = ClassUtils.get_filename_extension(full_path)

                if extension == '.mjpegx':
                    print(file)
                    print('Reprocessing ' + full_path + ' to mjpegx')
                    camera_number = ClassUtils.get_cam_number_from_path(full_path)
                    ClassMjpegConverter.convert_video_mjpegx(full_path, camera_number, instance_nn_pose)


if __name__ == '__main__':
    main()
