import numpy as np
import cv2


def main():
    """Running main function"""
    print('Initializing function')
    print('Reading video')

    cap = cv2.VideoCapture('test.mp4')
    period_ms = 200

    filter_mask = cv2.bgsegm.createBackgroundSubtractorMOG()
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = filter_mask.apply(frame)

        cv2.imshow('frame', fg_mask)
        if cv2.waitKey(period_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Bye')


main()
