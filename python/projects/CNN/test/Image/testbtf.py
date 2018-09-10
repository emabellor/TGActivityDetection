import os
from classutils import ClassUtils
from classdescriptors import ClassDescriptors
import json
import matplotlib.pyplot as plt
from tkinter import Tk
import cv2
from tkinter.filedialog import askopenfilename
from classopenpose import ClassOpenPose
import numpy as np

base1_folder = '/home/mauricio/Pictures/BTF/0/'
base2_folder = '/home/mauricio/Pictures/BTF/1/'

min_score = 0.05


def main():
    Tk().withdraw()

    instance_pose = ClassOpenPose()

    print('Initializing main function')
    list_files1 = os.listdir(base1_folder)
    hists_1 = read_hists(base1_folder, list_files1)

    list_files2 = os.listdir(base2_folder)
    hists_2 = read_hists(base2_folder, list_files2)

    cum_hists_1 = ClassDescriptors.get_cumulative_hists(hists_1)
    cum_hists_2 = ClassDescriptors.get_cumulative_hists(hists_2)

    # Processing img
    filename1 = '/home/mauricio/Pictures/Poses/bend_left/636453039344460032_1.jpg'
    filename2 = '/home/mauricio/Pictures/Poses/left_walk/636550795366450048_420.jpg'

    init_dir = '/home/mauricio/Pictures'
    options = {'initialdir': init_dir}
    filename1 = askopenfilename(**options)

    if not filename1:
        filename1 = '/home/mauricio/Pictures/2_1.jpg'

    ext1 = os.path.splitext(filename1)[1]
    if ext1 != '.jpg' and ext1 != '.jpeg':
        raise Exception('Extension1 is not jpg or jpeg')

    print('Loading filename 2')
    filename2 = askopenfilename(**options)

    if not filename2:
        filename2 = '/home/mauricio/Pictures/2_2.jpg'

    ext2 = os.path.splitext(filename2)[1]
    if ext2 != '.jpg' and ext2 != '.jpeg':
        raise Exception('Extension2 is not jpg or jpeg')

    image1 = cv2.imread(filename1)
    if image1 is None:
        raise Exception('Invalid image in filename {0}'.format(filename1))

    image2 = cv2.imread(filename2)
    if image2 is None:
        raise Exception('Invalid image in filename {0}'.format(filename2))

    is_json = True
    new_file_1 = filename1.replace('.jpeg', '.json')
    new_file_1 = new_file_1.replace('.jpg', '.json')

    new_file_2 = filename2.replace('.jpeg', '.json')
    new_file_2 = new_file_2.replace('.jpg', '.json')

    if not os.path.exists(new_file_1):
        print('File not found: {0}'.format(new_file_1))
        is_json = False

    if not os.path.exists(new_file_2):
        print('File not found: {0}'.format(new_file_2))
        is_json = False

    if not is_json:
        poses1 = instance_pose.recognize_image(image1)
        poses2 = instance_pose.recognize_image(image2)

        if len(poses1) != 1:
            raise Exception('Invalid len for pose1: {0}'.format(len(poses1)))
        if len(poses2) != 1:
            raise Exception('Invalid len for pose2: {0}'.format(len(poses2)))
        if not ClassUtils.check_vector_integrity_pos(poses1[0], min_score):
            raise Exception('Pose 1 not valid')
        if not ClassUtils.check_vector_integrity_pos(poses2[0], min_score):
            raise Exception('Pose 2 not valid')

        pose1 = poses1[0]
        pose2 = poses2[0]
    else:
        with open(new_file_1, 'r') as f:
            obj_json1 = json.loads(f.read())

        with open(new_file_2, 'r') as f:
            obj_json2 = json.loads(f.read())

        pose1 = obj_json1['vector']
        pose2 = obj_json2['vector']

        if not ClassUtils.check_vector_integrity_pos(pose1, min_score):
            raise Exception('Pose 1 not valid')
        if not ClassUtils.check_vector_integrity_pos(pose2, min_score):
            raise Exception('Pose 2 not valid')

    # Plotting
    # plot_histograms(hists1)

    # Showing first images without transformation
    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)

    upper1, lower1 = ClassDescriptors.process_colors_person(pose1, min_score, image1, decode_img=False)
    upper2, lower2 = ClassDescriptors.process_colors_person(pose2, min_score, image2, decode_img=False)

    print('Upper1 {0} - Lower1 {0}'.format(upper1, lower1))
    print('Upper2 {0} - Lower2 {0}'.format(upper2, lower2))

    print('Diff1: {0}'.format(ClassUtils.get_color_diff_rgb(upper1, upper2)))
    print('Diff2: {0}'.format(ClassUtils.get_color_diff_rgb(lower1, lower2)))

    cv2.imshow('main_window', np.hstack((image1, image2)))
    print('Press any key to continue')
    cv2.waitKey(0)

    # Perform image transformation
    image_tr = ClassDescriptors.transform_image(image2, cum_hists_2, cum_hists_1)
    upper1, lower1 = ClassDescriptors.process_colors_person(pose1, min_score, image1, decode_img=False)
    upper2, lower2 = ClassDescriptors.process_colors_person(pose2, min_score, image_tr, decode_img=False)

    print('Diff1: {0}'.format(ClassUtils.get_color_diff_rgb(upper1, upper2)))
    print('Diff2: {0}'.format(ClassUtils.get_color_diff_rgb(lower1, lower2)))

    cv2.imshow('main_window', np.hstack((image1, image_tr)))
    print('Press any key to continue')
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    print('Done!')


def read_hists(folder_name, list_files):
    hist_red = [0 for _ in range(256)]
    hist_green = [0 for _ in range(256)]
    hist_blue = [0 for _ in range(256)]

    for path in list_files:
        fullname = os.path.join(folder_name, path)

        ext = ClassUtils.get_filename_extension(fullname)

        if ext == '.json':
            with open(fullname, 'r') as f:
                dict_json_str = f.read()

            dict_json = json.loads(dict_json_str)
            hists = dict_json['hists']
            for i in range(256):
                hist_red[i] += hists[0][i]
                hist_green[i] += hists[0][i]
                hist_blue[i] += hists[0][i]

    return hist_red, hist_green, hist_blue


def plot_histograms(hist_list):
    red_hist_cum = hist_list[0]
    green_hist_cum = hist_list[1]
    blue_hist_cum = hist_list[2]

    # How to plot an histogram in python
    # Plot as bars
    # Refer to this link
    # https://stackoverflow.com/questions/43139771/plot-bar-by-x-and-y-coordinates
    x = [x for x in range(256)]

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].bar(x, red_hist_cum, edgecolor="k")
    ax[0, 1].bar(x, green_hist_cum, edgecolor="k")
    ax[1, 0].bar(x, blue_hist_cum, edgecolor="k")

    plt.show()


if __name__ == '__main__':
    main()
