from classutils import ClassUtils
from classdescriptors import ClassDescriptors
from classopenpose import ClassOpenPose
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import cv2


min_score = 0.05


def main():
    print('Initializing main function')

    Tk().withdraw()

    # Loading instances
    instance_pose = ClassOpenPose()

    # Loading filename
    init_dir = '/home/mauricio/Pictures/TestPlumb'
    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if not filename:
        filename = '/home/mauricio/Pictures/419.jpg'

    print('Filename: {0}'.format(filename))
    image = cv2.imread(filename)

    arr = instance_pose.recognize_image(image)
    valid_arr = list()

    for person_arr in arr:
        if ClassUtils.check_vector_integrity_pos(person_arr, min_score):
            valid_arr.append(person_arr)

    if len(valid_arr) != 1:
        raise Exception('Invalid len for arr: {0}'.format(len(valid_arr)))

    person_arr = valid_arr[0]
    ClassDescriptors.draw_pose(image, person_arr, min_score)

    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('main_window', image)

    print_distances(person_arr)

    cv2.waitKey()
    cv2.destroyAllWindows()

    print('Done!')


def print_distances(person_vector):
    print('Reading distances')

    # Draw plumb position
    print('Torso')
    print_points(person_vector[1], person_vector[8])

    print('Femur1')
    print_points(person_vector[9], person_vector[10])

    print('Femur2')
    print_points(person_vector[12], person_vector[13])


def print_points(point1, point2):
    if ClassUtils.check_point_integrity(point1, min_score) and \
            ClassUtils.check_point_integrity(point2, min_score):

        print('Point1: {0} - Point2: {1}'.format(point1, point2))
        distance_plumb = ClassUtils.get_euclidean_distance_pt(point1, point2)
        print('Distance plumb: {0}'.format(distance_plumb))


if __name__ == '__main__':
    main()
