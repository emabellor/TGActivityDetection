from datetime import datetime
from classutils import ClassUtils


def main():
    print('Initializing main function')

    date = datetime(2018, 2, 1, 12, 0, 0, 4563)
    print('Date: {0}'.format(date))
    ticks = ClassUtils.datetime_to_ticks(date)

    print('Ticks: {0}'.format(ticks))

    new_date = ClassUtils.ticks_to_datetime(ticks)
    print('New datetime: {0}'.format(new_date))

    print('Done!')


if __name__ == '__main__':
    main()
