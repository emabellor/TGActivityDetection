from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
from classopenpose import ClassOpenPose
from classutils import ClassUtils
from classdescriptors import ClassDescriptors
import numpy as np
import math


def main():
    print('Initializing main function')
    Tk().withdraw()

    # filename1 = '/home/mauricio/Pictures/Poses/temp/636550787632290048_419.jpg'
    # filename2 = '/home/mauricio/Pictures/Poses/walk_front/636550801813440000_424.jpg'
    filename1 = '/home/mauricio/Pictures/Poses/bend_left/636453039344460032_1.jpg'
    filename2 = '/home/mauricio/Pictures/Poses/left_walk/636550795366450048_420.jpg'

    print('Select first file')
    
    init_dir = '/home/mauricio/Pictures/Poses'
    options = {'initialdir': init_dir}
    filename1_tmp = askopenfilename(**options)

    if filename1_tmp:
        filename1 = filename1_tmp
    
    print('File selected: {0}'.format(filename1))
    print('Select second file')
    filename2_tmp = askopenfilename(**options)

    if filename2_tmp:
        filename2 = filename2_tmp
    
    print('File selected: {0}'.format(filename2))
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    instance_pose = ClassOpenPose()
    vectors1, img_draw1 = instance_pose.recognize_image_tuple(img1)
    vectors2, img_draw2 = instance_pose.recognize_image_tuple(img2)

    if len(vectors1) != 1:
        raise Exception('Invalid len for vector 1')

    if len(vectors2) != 1:
        raise Exception('Invalid len for vector 2')

    person_vector1 = vectors1[0]
    person_vector2 = vectors2[0]

    min_percent = 0.05

    if not ClassUtils.check_vector_integrity_pos(person_vector1, min_percent):
        raise Exception('Invalid integrity for vector 1')

    if not ClassUtils.check_vector_integrity_pos(person_vector2, min_percent):
        raise Exception('Invalid integrity for vector 2')

    colors1 = ClassDescriptors.process_colors(vectors1, min_percent, img1, decode_img=False)
    colors2 = ClassDescriptors.process_colors(vectors2, min_percent, img2, decode_img=False)

    upper1 = colors1[0][0]
    lower1 = colors1[1][0]
    color_diff1 = ClassUtils.get_color_diff_rgb(upper1, lower1)

    upper2 = colors2[0][0]
    lower2 = colors2[1][0]
    color_diff2 = ClassUtils.get_color_diff_rgb(upper2, lower2)

    diff_upper = ClassUtils.get_color_diff_rgb(upper1, upper2)
    diff_lower = ClassUtils.get_color_diff_rgb(lower1, lower2)
    diff_diff = math.fabs(color_diff1 - color_diff2)

    print('Diff upper: {0}'.format(diff_upper))
    print('Diff lower: {0}'.format(diff_lower))
    print('Diff diffs: {0}'.format(diff_diff))

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    cv2.imshow('main_window', np.hstack((img_draw1, img_draw2)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Done!')
    

if __name__ == '__main__':
    main()
