"""
Written by
Eder Mauricio Abello Rodriguez
"""
import mjpeg.mjpeghandler as mjpeg
import cv2
import numpy as np


def main():
    """Main Function"""
    path_video = 'video2.mjpeg'

    print('Init python script')
    number = ""
    while True:
        try:
            method = input('Select background subtraction method\n')
            number = int(method)

            if number >= 0 & number <= 6:
                break
            else:
                print('Number is not valid')
        except ValueError:
            print('Number is not valid - Try Again')

    filter_mask = ""

    if number == 0:
        filter_mask = cv2.createBackgroundSubtractorMOG2()
    elif number == 1:
        filter_mask = cv2.createBackgroundSubtractorKNN()
    elif number == 2:
        filter_mask = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif number == 3:
        filter_mask = cv2.bgsegm.createBackgroundSubtractorGMG()
    elif number == 4:
        filter_mask = cv2.bgsegm.createBackgroundSubtractorCNT()
    elif number == 5:
        filter_mask = cv2.bgsegm.createBackgroundSubtractorGSOC()
    else:
        filter_mask = cv2.bgsegm.createBackgroundSubtractorLSBP()

    print('Loading list of frames')
    video_list = mjpeg.get_frame_list(path_video)

    print('Creating instances')
    period_ms = 200

    print('Iterating over video list')
    for frame in video_list:
        frame_bin = frame['frame_bin']
        array_np = np.frombuffer(frame_bin, dtype=np.uint8)

        image = cv2.imdecode(array_np, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fg_mask = filter_mask.apply(gray)

        cv2.imshow('frame', fg_mask)
        if cv2.waitKey(period_ms) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print('Bye')


print('Calling Main Function')
main()
