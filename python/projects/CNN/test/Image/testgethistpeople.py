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
    option = input('Select 1 to hist by line - select 2 to hist by pose - 3 to compare images: ')

    if option == '1':
        hist_by_line()
    elif option == '2':
        hist_by_pose()
    elif option == '3':
        compare_images()
    else:
        raise Exception('Option not recognized!')


def hist_by_line():
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

    # Loading and drawing image
    width = ClassDescriptors.get_width_relation(pose)
    points1 = ClassDescriptors.get_points_by_line(image, pose[2], pose[3], width, draw=False)
    points2 = ClassDescriptors.get_points_by_line(image, pose[3], pose[4], width, draw=False)

    # Showing image
    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    ClassUtils.draw_pose(image, pose, min_score)
    cv2.imshow('main_window', image)

    print('Press any key to continue')
    cv2.waitKey()

    # Destroying all windows
    cv2.destroyAllWindows()

    print('Done!')


def hist_by_pose():
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


def compare_images():
    print('Comparing images')

    instance_pose = ClassOpenPose()

    items = ClassDescriptors.load_images_comparision_ext(instance_pose, min_score)

    list_points1 = items['listPoints1']
    list_points2 = items['listPoints2']

    hist1_np = np.array(list_points1)
    hist2_np = np.array(list_points2)

    clusters = 3
    clt1 = KMeans(n_clusters=clusters)
    clt1.fit(hist1_np)

    clt2 = KMeans(n_clusters=clusters)
    clt2.fit(hist2_np)

    print(clt1.cluster_centers_)
    print(clt2.cluster_centers_)

    # Compare elements
    for i in range(3):
        color1 = clt1.cluster_centers_[i]
        for j in range(3):
            color2 = clt2.cluster_centers_[j]
            diff = ClassUtils.get_color_diff_rgb(color1, color2)
            print(diff)

    hist1_norm = ClassDescriptors.centroid_histogram(clt1)
    print(hist1_norm)
    bar1 = plot_colors(hist1_norm, clt1.cluster_centers_)

    hist2_norm = ClassDescriptors.centroid_histogram(clt2)
    bar2 = plot_colors(hist2_norm, clt2.cluster_centers_)

    plt.figure()
    plt.axis("off")
    plt.imshow(np.vstack((bar1, bar2)))
    plt.show()

    print('Done1')


if __name__ == '__main__':
    main()

