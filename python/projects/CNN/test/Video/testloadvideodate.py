from classmjpegdate import ClassMjpegDate
from datetime import datetime
import logging

logger = logging.getLogger('Main')


def main():
    FORMAT = "%(asctime)s [%(name)-16.16s] [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    logger.info('Initializing main function')
    logger.info('Generating test')

    id_cam = 419
    instance = ClassMjpegDate(id_cam)
    date = datetime(2018, 2, 24, 14, 35, 0)
    frame = instance.load_frame(date)

    if frame is None:
        logger.error('There is no frame')
    else:
        image = frame[0]

        logger.info('frame len: {0}'.format(len(image)))


if __name__ == '__main__':
    main()
