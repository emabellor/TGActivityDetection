from tkinter.filedialog import askopenfilename
from tkinter import filedialog
from tkinter import Tk
import cv2
import os
from classopenpose import ClassOpenPose
from classdescriptors import ClassDescriptors
from classutils import ClassUtils
import matplotlib.pyplot as plt
import numpy as np
import json

min_score = 0.05

CNN_BASE_FOLDER = '/home/mauricio/CNN/Images'
SAVE_IMG_BASE_FOLDER = '/home/mauricio/Pictures/BTF/Examples'
CANDIDATES_FOLDER = '/home/mauricio/Pictures/BTF/Candidates'
EXAMPLES_FOLDER = '/home/mauricio/Pictures/BTF/Examples'


def main():
    Tk().withdraw()
    print('Initializing main function')

    option = input('Select 1 to test single - Select 2 to test BTF transformation, 3 test to adjust with '
                   'LAB adjustment - 4 to test lab with json imgs -  5 to test color eq - '
                   '6 to test color comparision - 7 to test full color comparision - '
                   '8 to view color histogram - 9 to test compare color hist - '
                   '10 to test compare color hist eq - 11 to reprocess folder: ')

    if option == '1':
        process_single()
    elif option == '2':
        process_btf()
    elif option == '3':
        process_lab_adj()
    elif option == '4':
        process_lab_adj_json()
    elif option == '5':
        process_test_color_eq()
    elif option == '6':
        test_color_compare()
    elif option == '7':
        test_full_color_compare()
    elif option == '8':
        test_view_histogram()
    elif option == '9':
        test_color_compare_hist()
    elif option == '10':
        test_color_compare_hist_eq()
    elif option == '11':
        re_process_folder()
    else:
        print('Option not recognized: {0}'.format(option))


def process_btf():
    print('Processing BTF transformation')

    # Loading instances
    instance_pose = ClassOpenPose()
    image1, image2, pose1, pose2 = ClassDescriptors.load_images_comparision(instance_pose, min_score)

    hists1 = ClassDescriptors.get_color_histograms(pose1, min_score, image1, decode_img=False)
    hists2 = ClassDescriptors.get_color_histograms(pose2, min_score, image2, decode_img=False)

    # Plotting
    # plot_histograms(hists1)

    # Showing first images without transformation
    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)

    upper1, lower1 = ClassDescriptors.process_colors_person(pose1, min_score, image1, decode_img=False)
    upper2, lower2 = ClassDescriptors.process_colors_person(pose2, min_score, image2, decode_img=False)

    print('Diff1: {0}'.format(ClassUtils.get_color_diff_rgb(upper1, upper2)))
    print('Diff2: {0}'.format(ClassUtils.get_color_diff_rgb(lower1, lower2)))

    cv2.imshow('main_window', np.hstack((image1, image2)))
    print('Press any key to continue')
    cv2.waitKey(0)

    # Perform image transformation
    image_tr = ClassDescriptors.transform_image(image2, hists2, hists1)
    upper1, lower1 = ClassDescriptors.process_colors_person(pose1, min_score, image1, decode_img=False)
    upper2, lower2 = ClassDescriptors.process_colors_person(pose2, min_score, image_tr, decode_img=False)

    print('Diff1: {0}'.format(ClassUtils.get_color_diff_rgb(upper1, upper2)))
    print('Diff2: {0}'.format(ClassUtils.get_color_diff_rgb(lower1, lower2)))

    cv2.imshow('main_window', np.hstack((image1, image_tr)))
    print('Press any key to continue')
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    print('Done!')


def process_single():
    print('Initializing process_single')

    # Loading instances
    instance_pose = ClassOpenPose()

    # Opening filename
    init_dir = '/home/mauricio/Pictures'
    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if not filename:
        filename = '/home/mauricio/Pictures/Poses/left_walk/636550788328999936_420.jpg'

    # Reading video extension
    extension = os.path.splitext(filename)[1]

    if extension != '.jpg' and extension != '.jpeg':
        raise Exception('Extension is not jpg or jpeg')

    # Opening filename
    image = cv2.imread(filename)

    # Loading image and array
    arr, processed_img = instance_pose.recognize_image_tuple(image)

    # Showing image to generate elements
    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)

    arr_pass = list()
    min_score = 0.05

    # Checking vector integrity for all elements
    # Verify there is at least one arm and one leg
    for elem in arr:
        if ClassUtils.check_vector_integrity_pos(elem, min_score):
            arr_pass.append(elem)

    if len(arr_pass) != 1:
        print('There is more than one person in the image')
        cv2.imshow('main_window', processed_img)
        cv2.waitKey(0)
    else:
        person_array = arr_pass[0]
        print('Person array: {0}'.format(person_array))
        generate_descriptors(person_array, image, processed_img, min_score)


def generate_descriptors(person_array, image, processed_img, min_score):
    results = ClassDescriptors.get_person_descriptors(person_array, min_score)
    print('Results: {0}'.format(results))

    # Generate color histograms
    hists = ClassDescriptors.get_color_histograms(person_array, min_score, image, decode_img=False)

    red_hist_cum = hists[0]
    green_hist_cum = hists[1]
    blue_hist_cum = hists[2]

    cv2.imshow('main_window', processed_img)
    cv2.waitKey(0)

    plot_histograms(red_hist_cum)

    print('Done!')


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


def process_lab_adj():
    print('Process lab adjustment')

    # Loading instances
    instance_pose = ClassOpenPose()

    # Loading images
    image1, image2, pose1, pose2 = ClassDescriptors.load_images_comparision(instance_pose, min_score)

    # Performing color comparision
    upper1, lower1 = ClassDescriptors.process_colors_person(pose1, min_score, image1, decode_img=False)
    upper2, lower2 = ClassDescriptors.process_colors_person(pose2, min_score, image2, decode_img=False)

    # Performing custom comparison first
    diff_upper = ClassUtils.get_color_diff_rgb(upper1, upper2)
    diff_lower = ClassUtils.get_color_diff_rgb(lower1, lower2)

    print('Diff upper: {0}'.format(diff_upper))
    print('Diff lower: {0}'.format(diff_lower))

    # Performing lab conversion and equal L values
    diff_upper_lum = ClassUtils.get_color_diff_rgb_lum(upper1, upper2)
    diff_lower_lum = ClassUtils.get_color_diff_rgb_lum(lower1, lower2)

    print('Diff upper lum: {0}'.format(diff_upper_lum))
    print('Diff lower lum: {0}'.format(diff_lower_lum))

    # Showing images
    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    cv2.imshow('main_window', np.hstack((image1, image2)))

    cv2.waitKey(0)

    print('Done!')


def process_lab_adj_json():
    print('Processing lab adj json!')
    return process_lab_adj()


def process_test_color_eq():
    print('Processing color eq')

    # Loading instances
    instance_pose = ClassOpenPose()

    im1, im2, pose1, pose2 = ClassDescriptors.load_images_comparision(instance_pose, min_score)

    upper1, lower1 = ClassDescriptors.process_colors_person(pose1, min_score, im1, decode_img=False)
    upper2, lower2 = ClassDescriptors.process_colors_person(pose2, min_score, im2, decode_img=False)

    img_size = 100
    image1 = np.zeros((img_size, img_size, 3), np.uint8)
    image2 = np.zeros((img_size, img_size, 3), np.uint8)
    image3 = np.zeros((img_size, img_size, 3), np.uint8)
    image4 = np.zeros((img_size, img_size, 3), np.uint8)

    # Generating elements in list
    cv2.namedWindow('aux_window', cv2.WND_PROP_AUTOSIZE)
    cv2.imshow('aux_window', np.hstack((im1, im2)))

    # BGR format -> Generating
    image1[:, :] = (upper1[2], upper1[1], upper1[0])
    image2[:, :] = (lower1[2], lower1[1], lower1[0])

    image3[:, :] = (upper2[2], upper2[1], upper2[0])
    image4[:, :] = (lower2[2], lower2[1], lower2[0])

    # Performing custom comparison first
    diff_upper = ClassUtils.get_color_diff_rgb(upper1, upper2)
    diff_lower = ClassUtils.get_color_diff_rgb(lower1, lower2)

    print('Diff upper: {0}'.format(diff_upper))
    print('Diff lower: {0}'.format(diff_lower))

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)

    cv2.imshow('main_window', np.hstack((np.vstack((image1, image2)), np.vstack((image3, image4)))))
    cv2.waitKey(0)

    # Performing lab conversion and equal L values
    diff_upper_lum = ClassUtils.get_color_diff_rgb_lum(upper1, upper2)
    diff_lower_lum = ClassUtils.get_color_diff_rgb_lum(lower1, lower2)

    # Transform point
    upper2_eq = ClassUtils.eq_lum_rgb_colors(upper1, upper2)
    lower2_eq = ClassUtils.eq_lum_rgb_colors(lower1, lower2)

    print('Diff upper lum: {0}'.format(diff_upper_lum))
    print('Diff lower lum: {0}'.format(diff_lower_lum))

    image3[:, :] = (upper2_eq[2], upper2_eq[1], upper2_eq[0])
    image4[:, :] = (lower2_eq[2], lower2_eq[1], lower2_eq[0])

    # Showing again image
    cv2.imshow('main_window', np.hstack((np.vstack((image1, image2)), np.vstack((image3, image4)))))
    cv2.waitKey(0)

    # Destroying all windows
    cv2.destroyAllWindows()

    print('Done!')


def test_color_compare():
    print('Test color comparision')

    print('Loading image comparision')
    # Loading instances
    instance_pose = ClassOpenPose()

    # Avoid to open two prompts
    obj_img = ClassDescriptors.load_images_comparision_ext(instance_pose, min_score, load_one_img=True)
    hist_pose1 = obj_img['listPoints1']

    list_process = list()

    # Iterating in examples folder
    for root, _, files in os.walk(EXAMPLES_FOLDER):
        for file in files:
            full_path = os.path.join(root, file)
            extension = ClassUtils.get_filename_extension(full_path)

            if extension == '.jpg':
                list_process.append(full_path)

    # Sorting list
    list_process.sort()

    list_result = list()
    score_max_pt = -1

    for full_path in list_process:
        print('Processing file: {0}'.format(full_path))
        json_path = full_path.replace('.jpg', '.json')

        with open(json_path, 'r') as f:
            obj_json = json.loads(f.read())

        his_pose2 = obj_json['histPose']
        diff = ClassDescriptors.get_kmeans_diff(hist_pose1, his_pose2)
        print('Diff {0}'.format(diff))

        if diff <= 15:
            res = True
        else:
            res = False

        list_result.append({
            'filename': ClassUtils.get_filename_no_extension(full_path),
            'score': res
        })

    # list_result.sort(key=lambda x: x['score'])
    print('Printing list result')
    print(json.dumps(list_result, indent=2))
    print('min_score: {0}'.format(score_max_pt))

    print('Done!')


def test_color_compare_hist(perform_eq=False):
    print('Test color comparision')

    print('Loading image comparision')
    # Loading instances
    instance_pose = ClassOpenPose()

    ignore_json_color = False
    if perform_eq:
        ignore_json_color = True

    # Avoid to open two prompts
    obj_img = ClassDescriptors.load_images_comparision_ext(instance_pose, min_score, load_one_img=True,
                                                           perform_eq=perform_eq, ignore_json_color=ignore_json_color)

    # Extract color comparision from image
    hist1 = obj_img['listPoints1']

    # Generating examples folder
    list_process = list()
    for root, _, files in os.walk(EXAMPLES_FOLDER):
        for file in files:
            full_path = os.path.join(root, file)
            extension = ClassUtils.get_filename_extension(full_path)

            if extension == '.jpg':
                list_process.append(full_path)

    # Sorting list
    list_process.sort()

    """
    list_result = list()
    score_max_pt = -1
    """

    for full_path in list_process:
        print('Processing file: {0}'.format(full_path))
        json_path = full_path.replace('.jpg', '.json')

        with open(json_path, 'r') as f:
            obj_json = json.loads(f.read())

        image2 = cv2.imread(full_path)

        if perform_eq:
            image2 = ClassUtils.equalize_hist(image2)

        if 'vector' in obj_json:
            pose2 = obj_json['vector']
        elif 'vectors' in obj_json:
            pose2 = obj_json['vectors']
        else:
            raise Exception('Invalid vector property for vector custom')

        hist2 = ClassDescriptors.get_points_by_pose(image2, pose2, min_score)

        diff = ClassDescriptors.get_kmeans_diff(hist1, hist2)

        # Getting mean y from image 2 - discarding purposes
        pt1, pt2 = ClassUtils.get_rectangle_bounds(pose2, min_score)
        image2_crop = image2[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        image2_ycc = cv2.cvtColor(image2_crop, cv2.COLOR_BGR2YCrCb)
        mean_y = np.mean(image2_ycc[:, :, 0])

        print('Diff color: {0} - Mean y: {1}'.format(diff, mean_y))

    """
    list_result.sort(key=lambda x: x['score'])
    print('Printing list result')
    print(list_result)
    print('min_score: {0}'.format(score_max_pt))
    """

    print('Done!')


def test_full_color_compare():
    print('Process lab adjustment')

    # Loading instances
    instance_pose = ClassOpenPose()

    # Loading images
    image1, image2, pose1, pose2 = ClassDescriptors.load_images_comparision(instance_pose, min_score)

    # Performing color comparision
    upper1, lower1 = ClassDescriptors.process_colors_person(pose1, min_score, image1, decode_img=False)
    upper2, lower2 = ClassDescriptors.process_colors_person(pose2, min_score, image2, decode_img=False)

    # Performing custom comparison first
    diff_upper, diff_lower, delta = ClassUtils.compare_colors(upper1, upper2, lower1, lower2)

    print('Diff upper: {0}'.format(diff_upper))
    print('Diff lower: {0}'.format(diff_lower))
    print('Delta: {0}'.format(delta))

    # Showing images
    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    cv2.imshow('main_window', np.hstack((image1, image2)))

    cv2.waitKey(0)

    print('Done!')


def test_view_histogram():
    print('Test view histogram')

    # Loading instances
    instance_pose = ClassOpenPose()

    # Loading images
    image1, _, pose1, _ = ClassDescriptors.load_images_comparision(instance_pose, min_score, load_one_img=True)

    # Drawing poses
    ClassDescriptors.draw_pose(image1, pose1, min_score)

    # Showing image
    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    cv2.imshow('main_window', image1)

    print('Press a key to continue')
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    print('Done!')


def test_color_compare_hist_eq():
    print('Initialize color compare hist eq')
    perform_eq = True
    test_color_compare_hist(perform_eq)


def re_process_folder():
    print('Init folder processing')

    init_dir = '/home/mauricio/Pictures/BTF'
    options = {'initialdir': init_dir}

    folder = filedialog.askdirectory(**options)

    if folder is None:
        raise Exception('Folder not selected!')

    files = os.listdir(folder)

    for file in files:
        full_path = os.path.join(folder, file)
        ext = ClassUtils.get_filename_extension(full_path)

        if ext == '.jpg':
            print('Processing file: {0}'.format(full_path))

            file_json = ClassUtils.get_filename_no_extension(full_path) + '.json'

            with open(file_json, 'r') as f:
                json_txt = f.read()

            json_data = json.loads(json_txt)

            if 'vectors' in json_data:
                vectors = json_data['vectors']
            elif 'vector' in json_data:
                vectors = json_data['vector']
            else:
                raise Exception('Vector not found!')

            img_cv = cv2.imread(full_path)

            param = ClassDescriptors.get_person_descriptors(vectors, min_score, image=img_cv, decode_img=False)
            with open(file_json, 'w') as f:
                f.write(json.dumps(param, indent=2))

    print('Done!')


if __name__ == '__main__':
    main()
