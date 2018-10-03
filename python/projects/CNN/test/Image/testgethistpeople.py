from classutils import ClassUtils
from classdescriptors import ClassDescriptors
from classopenpose import ClassOpenPose
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time

min_score = 0.05


def main():
    Tk().withdraw()
    print('Initializing main function')
    option = input('Select 1 to hist by line - select 2 to hist by pose - 3 to compare images -'
                   ' 4 to perform image equalization - 5 to get hist by pose eq -'
                   ' 6 to compare images eq - 7 to get image contrast - 8 to get pose contrast - '
                   ' 9 to get pose lum - 10 to test draw pose img: ')

    if option == '1':
        hist_by_line()
    elif option == '2':
        hist_by_pose()
    elif option == '3':
        compare_images()
    elif option == '4':
        eq_image()
    elif option == '5':
        get_hist_by_pose_eq()
    elif option == '6':
        compare_images_eq()
    elif option == '7':
        get_image_contrast()
    elif option == '8':
        get_pose_contrast()
    elif option == '9':
        get_pose_lum()
    elif option == '10':
        draw_pose_img()
    else:
        raise Exception('Option not recognized!')


def hist_by_line():
    print('Initializing main function')
    raise Exception('Not implemented! Use hist by pose')


def hist_by_pose(perform_eq=False):
    print('Initializing main function')

    # Initializing instances
    instance_pose = ClassOpenPose()

    # Loading image
    init_dir = '/home/mauricio/Pictures'
    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if not filename:
        filename = '/home/mauricio/Pictures/BTF/0/419-2018022414021519500346500000.jpg'

    image = cv2.imread(filename)

    # Reading skeletons
    poses = instance_pose.recognize_image(image)

    if len(poses) != 1:
        raise Exception('Invalid len for poses: {0}'.format(len(poses)))

    if not ClassUtils.check_vector_integrity_pos(poses[0], min_score):
        raise Exception('Invalid vector integrity')

    pose = poses[0]

    # Perform histogram equalization if flag is activated
    if perform_eq:
        image = ClassUtils.equalize_hist(image)

    # Loading and drawing image
    start = time.time()
    points = ClassDescriptors.get_points_by_pose(image, pose, min_score, draw=True)
    end = time.time()
    print('Elapsed: {0}'.format(end - start))
    points_np = np.array(points)

    max_clusters = 3
    min_clusters = 3
    clt = get_best_clt(points_np, max_clusters, min_clusters)

    # Showing image
    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    # ClassUtils.draw_pose(image, pose, min_score)
    cv2.imshow('main_window', image)

    print('Press any key to continue')
    cv2.waitKey()

    # Destroying all windows
    cv2.destroyAllWindows()

    # Plotting
    hist = ClassDescriptors.centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)

    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()

    print('Done!')


def get_best_clt(points_np, max_clusters, min_clusters):
    clusters = max_clusters
    clt = KMeans(n_clusters=clusters)
    for c in range(max_clusters, min_clusters - 1, -1):
        clusters = c
        clt = KMeans(n_clusters=clusters)
        clt.fit(points_np)

        valid = True
        for i in range(len(clt.cluster_centers_) - 1):
            for j in range(i + 1, len(clt.cluster_centers_)):
                diff = ClassUtils.get_color_diff_rgb(clt.cluster_centers_[i], clt.cluster_centers_[j])

                if diff < 10:
                    valid = False
                    break

            if not valid:
                break

        if valid:
            break

    print('Selected clusters: {0}'.format(clusters))
    return clt


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


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


def compare_images(perform_eq=False):
    print('Comparing images')

    instance_pose = ClassOpenPose()

    ignore_json_color = False
    if perform_eq:
        ignore_json_color = True

    draw_points = True
    items = ClassDescriptors.load_images_comparision_ext(instance_pose, min_score,
                                                         perform_eq=perform_eq,
                                                         ignore_json_color=ignore_json_color,
                                                         draw_points=draw_points)

    list_points1 = items['listPoints1']
    list_points2 = items['listPoints2']
    image1 = items['image1']
    image2 = items['image2']

    hist1_np = np.array(list_points1)
    hist2_np = np.array(list_points2)

    clusters = 3
    clt1 = KMeans(n_clusters=clusters)
    clt1.fit(hist1_np)

    clt2 = KMeans(n_clusters=clusters)
    clt2.fit(hist2_np)

    print(clt1.cluster_centers_)
    print(clt2.cluster_centers_)

    diff = ClassDescriptors.get_kmeans_diff(list_points1, list_points2)
    print('Diff images: {0}'.format(diff))

    # Add image showing before show histograms
    image_to_show = np.hstack((image1, image2))
    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('main_window', image_to_show)

    cv2.waitKey()
    cv2.destroyAllWindows()

    hist1_norm = ClassDescriptors.centroid_histogram(clt1)
    bar1 = plot_colors(hist1_norm, clt1.cluster_centers_)

    hist2_norm = ClassDescriptors.centroid_histogram(clt2)
    bar2 = plot_colors(hist2_norm, clt2.cluster_centers_)

    plt.figure()
    plt.axis("off")
    plt.imshow(np.vstack((bar1, bar2)))
    plt.show()

    print('Done1')


def eq_image():
    print('Performing image equalization')

    # Asking for image
    init_dir = '/home/mauricio/Pictures'
    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if not filename:
        filename = '/home/mauricio/Pictures/BTF/0/419-2018022414021519500346500000.jpg'

    image = cv2.imread(filename)

    # Performing image equalization in second option
    image_eq = ClassUtils.equalize_hist(image)

    # Showing image and image eq
    image_to_show = np.hstack((image, image_eq))

    # Showing eq image
    cv2.imshow('main_window', image_to_show)
    cv2.waitKey()

    # Destroying all windows
    cv2.destroyAllWindows()
    print('Done!')


def get_hist_by_pose_eq():
    print('Initializing get hist by pose eq')

    perform_eq = True
    hist_by_pose(perform_eq=perform_eq)


def compare_images_eq():
    print('Initializing compare images eq')

    perform_eq = True
    compare_images(perform_eq=perform_eq)


def get_image_contrast():
    # To check histogram
    # Please refer to this link
    # https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
    print('Getting image contrast')

    # Generating instances
    instance_pose = ClassOpenPose()

    # Performing image contrast
    items = ClassDescriptors.load_images_comparision_ext(instance_pose, min_score, load_one_img=True)

    image1 = items['image1']  # type: np.ndarray

    # Calculate RMS contrast of each histogram
    # Please refer to this link
    # https://en.wikipedia.org/wiki/Contrast_(vision)#RMS_contrast

    r_std = np.std(image1[:, :, 2])
    g_std = np.std(image1[:, :, 1])
    b_std = np.std(image1[:, :, 0])

    print('Read mean: {0}'.format(np.mean(image1[:, :, 0])))

    print('Red std: {0}'.format(r_std))
    print('Blue std: {0}'.format(g_std))
    print('Green std: {0}'.format(b_std))

    avg_std = (r_std + g_std + b_std) / 3
    print('Average contrast: {0}'.format(avg_std))

    print('Done!')


def get_pose_contrast():
    print('Getting pose contrast!')

    # Generating instances
    instance_pose = ClassOpenPose()

    items = ClassDescriptors.load_images_comparision_ext(instance_pose, min_score, load_one_img=True)
    image1 = items['image1']
    pose1 = items['pose1']

    # Get rectangle limits
    pt1, pt2 = ClassUtils.get_rectangle_bounds(pose1, min_score)

    # Cropping using numpy slicing
    image1_crop = image1[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    hist_b = cv2.calcHist([image1_crop], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image1_crop], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image1_crop], [2], None, [256], [0, 256])

    # Showing cropped image
    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('main_window', image1_crop)
    cv2.waitKey(0)

    # Calculating std from image
    r_std = np.std(image1_crop[:, :, 2])
    g_std = np.std(image1_crop[:, :, 1])
    b_std = np.std(image1_crop[:, :, 0])

    # Calculating mean from image
    r_mean = np.mean(image1_crop[:, :, 2])
    g_mean = np.mean(image1_crop[:, :, 1])
    b_mean = np.mean(image1_crop[:, :, 0])

    print('Mean values')
    print('Red std: {0}'.format(r_std))
    print('Green std: {0}'.format(g_std))
    print('Blue std: {0}'.format(b_std))
    print('--------')
    print('Std values')
    print('Red mean: {0}'.format(r_mean))
    print('Green mean: {0}'.format(g_mean))
    print('Blue mean: {0}'.format(b_mean))

    print('--------')
    print('Plotting histograms')
    # Print histogram to checking elements
    plot_histograms([hist_r, hist_g, hist_b])

    avg_std = (r_std + g_std + b_std) / 3
    print('Average std: {0}'.format(avg_std))
    cv2.destroyAllWindows()

    print('--------')
    print('Done!')


def get_pose_lum():
    print('Initializing getting pose lum')

    # Generating instances
    instance_pose = ClassOpenPose()

    items = ClassDescriptors.load_images_comparision_ext(instance_pose, min_score, load_one_img=True)
    image1 = items['image1']
    pose1 = items['pose1']

    pt1, pt2 = ClassUtils.get_rectangle_bounds(pose1, min_score)

    # Using numpy slicing
    image1_crop = image1[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    # Showing cropped image
    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('main_window', image1_crop)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Converting to YCbCR
    image1_lum = cv2.cvtColor(image1_crop, cv2.COLOR_BGR2YCrCb)

    mean_y = np.mean(image1_lum[:, :, 0])
    std_y = np.std(image1_lum[:, :, 0])

    print('Mean y: {0}'.format(mean_y))
    print('Std y: {0}'.format(std_y))

    print('--------')
    print('Done!')


def draw_pose_img():
    print('Initializing draw pose image')

    # Generating elements
    instance_pose = ClassOpenPose()

    items = ClassDescriptors.load_images_comparision_ext(instance_pose, min_score, load_one_img=True)
    pose_base = items['pose1']
    pose1 = items['transformedPoints1']
    torso_pixels = 100

    pose1 = ClassDescriptors.re_scale_pose_transformed(pose1, torso_pixels, min_score)
    image_pose = ClassDescriptors.draw_pose_image(pose1, min_score, is_transformed=True)
    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)

    cv2.imshow('main_window', image_pose)
    cv2.waitKey()

    # Generate pose mirroed
    pose1_mirrored = ClassDescriptors.mirror_pose_transformed(pose1)
    image_pose_mirrored = ClassDescriptors.draw_pose_image(pose1_mirrored, min_score, is_transformed=True)

    cv2.imshow('main_window', image_pose_mirrored)
    cv2.waitKey()

    cv2.destroyAllWindows()
    print('Done!')


if __name__ == '__main__':
    main()

