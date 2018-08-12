from classutils import ClassUtils


def main():
    print('Initializing main function')
    print('Loading elements')

    list_elems = [[1, 2, 3], [4, 5, 6]]
    list_flat = ClassUtils.get_flat_list(list_elems)
    print('list_flat: {0}'.format(list_flat))

    print('Done!')


if __name__ == '__main__':
    main()
