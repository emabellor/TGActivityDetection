from classpeoplereid import ClassPeopleReId


def main():
    print('Initializing main function')

    # Creating people reid
    person1 = ClassPeopleReId([], [], 0, 0, 1)
    person2 = ClassPeopleReId([], [], 0, 0, 2)

    print('Color person 1: {0}'.format(person1.get_rgb_color()))
    print('Color person 2: {0}'.format(person2.get_rgb_color()))

    print('Checking random variables')
    print('Color person 1: {0}'.format(person1.get_rgb_color()))
    print('Color person 2: {0}'.format(person2.get_rgb_color()))

    print('Done!')


if __name__ == '__main__':
    main()
