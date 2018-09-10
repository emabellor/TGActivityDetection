import numpy as np
import cv2
from classutils import ClassUtils
from colormath.color_objects import AdobeRGBColor, LabColor, HSVColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


def main():
    print('Initialize main function')
    selection = input('Press 1 to do color analysis - 2 to test Lab space - 3 to test conversion bgr: ')

    if selection == '1':
        do_color_analysis()
    elif selection == '2':
        test_lab_space()
    elif selection == '3':
        test_conversion_bgr()
    else:
        print('Command not identified')


def do_color_analysis():
    print('Initializing main function')

    print('HSV analysis')
    pt1 = np.array([[[129, 120, 116]]], dtype=np.uint8)
    pt2 = np.array([[[99, 101, 101]]], dtype=np.uint8)

    diff = ClassUtils.get_color_diff_rgb([129, 120, 116], [99, 101, 101])
    print(diff)

    hsv1 = cv2.cvtColor(pt1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(pt2, cv2.COLOR_BGR2HSV)

    hsv2[0, 0, 2] = 129
    pt_con = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    print(pt1)
    print(pt2)
    print(pt_con)

    diff = ClassUtils.get_color_diff_rgb(to_rgb_arr(pt1), to_rgb_arr(pt_con))
    print(diff)

    print('LAB analysis')
    lab1 = cv2.cvtColor(pt1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(pt2, cv2.COLOR_BGR2LAB)

    lab2[0, 0, 0] = 129
    pt_con = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    print(pt1)
    print(pt2)
    print(pt_con)

    diff = ClassUtils.get_color_diff_rgb(to_rgb_arr(pt1), to_rgb_arr(pt_con))
    print(diff)


def test_lab_space():
    print('Performing test lab space')
    color1_bgr = [128, 130, 150]

    color1 = sRGBColor(color1_bgr[2] / 255, color1_bgr[1] / 255, color1_bgr[0] / 255)
    color1_lab = convert_color(color1, LabColor)

    print(color1_lab)
    pt_cv = np.array([[color1_bgr]], dtype=np.uint8)
    print(type(pt_cv))
    print(pt_cv)

    lab1 = cv2.cvtColor(pt_cv, cv2.COLOR_BGR2LAB)
    print(lab1)


def test_conversion_bgr():
    print('Performing test conversion')

    color1_bgr = [128, 130, 150]
    color2_bgr = [50, 50, 200]

    pt_cv1 = np.array([[color1_bgr]], dtype=np.uint8)
    pt_cv2 = np.array([[color2_bgr]], dtype=np.uint8)

    diff_rgb = ClassUtils.get_color_diff_bgr(pt_cv1, pt_cv2)

    lab1 = cv2.cvtColor(pt_cv1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(pt_cv2, cv2.COLOR_BGR2LAB)

    diff_lab = ClassUtils.get_color_diff_lab(lab1, lab2)
    print('Diff rgb: {0}'.format(diff_rgb))
    print('Diff lab: {0}'.format(diff_lab))
    print('Done!')


def to_rgb_arr(pt: np.ndarray):
    return [int(pt[0, 0, 2]), int(pt[0, 0, 1]), int(pt[0, 0, 0])]


if __name__ == '__main__':
    main()
