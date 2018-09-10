from classpeoplereid import ClassPeopleReId
import time


def main():
    print('Initializing main function')
    person1 = ClassPeopleReId([], [0, 0, 0], [1, 1, 1], [255, 255, 255], [255, 255, 255], 1)
    person2 = ClassPeopleReId([], [0, 0, 0], [1, 1, 1], [255, 255, 255], [255, 255, 255], 2)

    print('Waiting some time...')
    time.sleep(1)

    print('Performing assignment')
    person3 = person2

    print('Updating')
    person3.update_values_from_person(person1)

    print('Time person 2: {0}'.format(person2.last_date))
    print('Time person 3: {0}'.format(person3.last_date))

    print('Done!')


if __name__ == '__main__':
    main()
