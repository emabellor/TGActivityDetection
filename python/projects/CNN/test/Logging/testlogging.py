import logging


def main():
    FORMAT = "%(asctime)s [%(name)-12.12s] [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    logger = logging.getLogger('ClassName')
    logger.info('Hola')
    logger.debug('Test logging')
    logger.error('Test error')


if __name__ == '__main__':
    main()
