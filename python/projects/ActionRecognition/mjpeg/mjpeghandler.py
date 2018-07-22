"""
Written by
Eder Mauricio Abello Rodriguez
"""
import struct


def get_frame_list(video_path):
    """Read file and return a list of mjpeg frames with date"""
    print('videoPath: ' + video_path)
    file_instance = open(video_path, 'rb')

    frame_list = []
    while True:
        size_bin = file_instance.read(4)

        if len(size_bin) == 0:
            print('End of file')
            break

        if len(size_bin) != 4:
            print('Error reading size')
            break

        size = struct.unpack('<i', size_bin)
        size = size[0]

        date_bin = file_instance.read(8)
        if len(date_bin) != 8:
            print('Error reading date')
            break

        date = struct.unpack('<q', date_bin)
        date = date[0]

        frame_bin = file_instance.read(size)
        if len(frame_bin) != size:
            print('Error reading frame')
            break

        new_item = {
            'date': date,
            'frame_bin': frame_bin
        }
        frame_list.append(new_item)

    print('Process executed')
    print('Total list: ', len(frame_list))
    return frame_list
