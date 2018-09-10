def main():
    print('Initializing main function')
    arr = [0, 1.2, 2, 3, 4]

    print('Array slice 0: {0}'.format(arr[:0]))
    print('Array slice 1: {0}'.format(arr[:1]))

    index = 2
    print('Array slice 2: {0}'.format(arr[:index+1]))

    print('Sum array slice 2: {0}'.format(sum(arr[:index+1])))
    print('Done!')


if __name__ == '__main__':
    main()
