from classutils import ClassUtils


def main():
    print('Initializing main function')

    ticks = 636764878330000000
    print('Converting ticks: {0}'.format(ticks))

    converted_date = ClassUtils.ticks_to_datetime(ticks)
    print('Date converted: {0}'.format(converted_date))

    print('Done!')


if __name__ == '__main__':
    main()

