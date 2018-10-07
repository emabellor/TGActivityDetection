from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json
import numpy as np
from classutils import ClassUtils
from classnn import ClassNN
import os
import cv2
import math

cnn_image_width = 28
cnn_image_height = 28


def main():
    Tk().withdraw()

    print('Initializing main function')
    res = input('Select option: 1 to test CNN image generation, '
                '2 to CNN image generation folder: ')

    if res == '1':
        cnn_image_generation()
    elif res == '2':
        cnn_image_generation_folder()
    else:
        raise Exception('Option not recognized')


def create_cnn_image_pose(param, instance_nn):
    list_poses = param['listPoses']
    max_confidence = 1

    image_height = 8
    image_width = len(list_poses)

    image_np = np.zeros((image_height, image_width), dtype=np.uint8)

    for index_pose, pose in enumerate(list_poses):
        angles = pose['angles']
        transformed_points = pose['transformedPoints']

        list_desc = list()
        list_desc += angles
        list_desc += ClassUtils.get_flat_list(transformed_points)

        list_desc_np = np.asanyarray(list_desc, dtype=np.float)

        res = instance_nn.predict_model_fast(list_desc_np)

        probabilities = res['probabilities']
        for index, value in enumerate(probabilities):
            pixel_value = int(value * 255 / max_confidence)
            image_np[index, index_pose] = pixel_value

    # Resizing image
    image_res = cv2.resize(image_np, (cnn_image_height, cnn_image_width))
    return image_res


def create_cnn_image_angles(param):
    # For now
    # Only consider transformed points

    list_poses = param['listPoses']

    min_angle = 0
    max_angle = math.pi

    # Last item - movement vector
    # Total elements: 27
    image_height = len(list_poses[0]['angles'])
    image_width = len(list_poses)

    image_np = np.zeros((image_height, image_width), dtype=np.uint8)

    for index_pose, pose in enumerate(list_poses):
        points = pose['angles']
        for index, item in enumerate(points):
            delta = math.pi

            if item < min_angle or item > max_angle:
                raise Exception('Invalid angle: {0}'.format(item))

            pixel_value = int((item - min_angle) * 255 / delta)
            image_np[index, index_pose] = pixel_value

    # Resizing image
    image_res = cv2.resize(image_np, (cnn_image_height, cnn_image_width))

    # Return created image
    return image_res


def create_cnn_image_pos(param):
    # For now
    # Only consider transformed points

    list_poses = param['listPoses']

    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    first = True

    for index, pose in enumerate(list_poses):
        points = pose['transformedPoints']

        for point in points:
            if first:
                min_x = point[0]
                min_y = point[1]
                max_x = min_x
                max_y = min_y
                first = False

            if point[0] < min_x:
                min_x = point[0]
            if point[1] < min_y:
                min_y = point[1]
            if point[0] > max_x:
                max_x = point[0]
            if point[1] > max_y:
                max_y = point[1]

    # Last item - movement vector
    # Total elements: 27
    image_height = len(ClassUtils.get_flat_list(list_poses[0]['transformedPoints']))
    image_width = len(list_poses)

    image_np = np.zeros((image_height, image_width), dtype=np.uint8)

    for index_pose, pose in enumerate(list_poses):
        points = pose['transformedPoints']
        vector_points = ClassUtils.get_flat_list(points)

        for index, item in enumerate(vector_points):
            if index % 2 == 0:
                min_value = min_x
                max_value = max_x
            else:
                min_value = min_y
                max_value = max_y

            delta = max_value - min_value
            if delta != 0:
                pixel_value = int((item - min_value) * 255 / delta)
            else:
                pixel_value = 0

            image_np[index, index_pose] = pixel_value

    # Resizing image
    image_res = cv2.resize(image_np, (cnn_image_height, cnn_image_width))

    # Return created image
    return image_res


def cnn_image_generation():
    print('Loading image generation')

    # Loading filename 1
    init_dir = '/home/mauricio/CNN/'

    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if filename is None:
        raise Exception('Filename not selected!!!')

    with open(filename, 'r') as f:
        json_txt = f.read()

    data_json = json.loads(json_txt)

    image_res = create_cnn_image_pos(data_json)

    cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
    cv2.imshow('main_window', image_res)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print('Done generating image')


def cnn_image_generation_folder():
    # Initializing instance nn
    classes_number = 8
    hidden_layers = 40

    list_folders = list()
    list_folders.append(ClassUtils.cnn_class_folder)
    instance_nn = ClassNN(ClassNN.model_dir_pose, classes_number, hidden_layers)

    # File walk
    for folder in list_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                extension = ClassUtils.get_filename_extension(full_path)

                if extension == '.json':
                    print('Processing: {0}'.format(full_path))

                    if 'ori' in full_path:
                        raise Exception('Full path contains ori folder!')

                    with open(full_path, 'r') as f:
                        json_txt = f.read()

                    json_data = json.loads(json_txt)

                    # All image generation
                    image_name_pos = ClassUtils.get_filename_no_extension(full_path) + '_p.jpg'
                    image_name_angle = ClassUtils.get_filename_no_extension(full_path) + '_a.jpg'
                    image_name_pose = ClassUtils.get_filename_no_extension(full_path) + '_s.jpg'

                    image_res_pos = create_cnn_image_pos(json_data)
                    image_res_angle = create_cnn_image_angles(json_data)
                    image_res_pose = create_cnn_image_pose(json_data, instance_nn)

                    print('Writing image pos: {0}'.format(image_name_pos))
                    cv2.imwrite(image_name_pos, image_res_pos)

                    print('Writing image angle: {0}'.format(image_name_angle))
                    cv2.imwrite(image_name_angle, image_res_angle)

                    print('Writing image pose: {0}'.format(image_name_pose))
                    cv2.imwrite(image_name_pose, image_res_pose)

    print('Done!')


if __name__ == '__main__':
    main()

