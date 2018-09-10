from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from classmjpegreader import ClassMjpegReader
import cv2
import numpy as np
import os


def main():
    print('Initializing main function')

    print('Select file to open')
    Tk().withdraw()

    init_dir = '/home/mauricio/Videos/Oviedo/2018-09-04/'
    filename, list_frames = load_video_info(init_dir)

    # Creating window to show frames
    print('Press any button to continue, esc to quit, s to save, c to change file')

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)

    index = 0
    while True:
        frame_info = list_frames[index]
        frame_bin = frame_info[0]

        image_np = np.frombuffer(frame_bin, dtype="int32")
        frame = cv2.imdecode(image_np, cv2.IMREAD_ANYCOLOR)

        print('Showing frame {0} of {1}'.format(index, len(list_frames)))
        cv2.imshow('main_window', frame)

        key = cv2.waitKey(0)
        print('Key pressed: {0}'.format(key))

        if key == 52:
            # Left arrow
            index -= 1
            if index < 0:
                index = 0
        elif key == 54:
            # Right arrow
            index += 1
            if index == len(list_frames):
                index = len(list_frames) - 1
        elif key == 27:
            # Esc
            break
        elif key == 115:
            # s
            print('Saving image')

            init_dir = '/home/mauricio/Pictures/Poses'
            options = {
                'initialdir': init_dir,
                'defaultextension': '.jpg'
            }
            file_image = asksaveasfilename(**options)

            if not file_image:
                print('Filename not selected')
            else:
                print('Saving image in {0}'.format(file_image))
                cv2.imwrite(file_image, frame)
                # Bug OpenCV
                cv2.destroyAllWindows()
                cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)

        elif key == 99:
            cv2.destroyAllWindows()
            dir_name = os.path.dirname(filename)
            print('Last processed video: {0}'.format(filename))
            filename, list_frames = load_video_info(dir_name)

            # Creating window to show frames
            print('Press any button to continue, esc to quit, s to save, c to change file')
            cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
            index = 0
        else:
            print('Key not selected')

    print('Done!')


def load_video_info(init_dir):
    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if not filename:
        raise Exception ('File not selected')

    print('Reading video extension')
    extension = os.path.splitext(filename)[1]

    if extension == '.mjpeg':
        list_frames = ClassMjpegReader.process_video(filename)
    elif extension == '.mjpegx':
        list_frames = ClassMjpegReader.process_video_mjpegx(filename)
    else:
        raise Exception('Extension not supported: {0}'.format(extension))

    return filename, list_frames


if __name__ == '__main__':
    main()
