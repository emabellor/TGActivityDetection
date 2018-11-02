import cv2
from classutils import ClassUtils
import numpy as np
import time

image_width = 800
image_height = 600
min_x = -200
max_x = 1000
min_y = -800
max_y = 800

flag_down = False

point_init_img = [0, 0]
point_end_img = [0, 0]
list_object_points = []

# Creating image into list
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
image[:, :] = (255, 255, 255)


def main():
    global list_object_points
    print('Initializing main function')

    list_cams = [419, 420, 421, 428, 429, 430]

    # Loading files
    for cam in list_cams:
        cam_str = str(cam)

        calib_params = ClassUtils.load_cam_calib_params(cam_str)
        object_points = calib_params['objectPoints']

        center_points = calib_params['centerPoints']
        angle_degrees = calib_params['angleDegrees']

        if angle_degrees == 0:
            for point in object_points:
                point[0] += center_points[0]
                point[1] += center_points[1]
        elif angle_degrees == 180:
            for point in object_points:
                point[0] = center_points[0] - point[0]
                point[1] = center_points[1] - point[1]
        else:
            raise Exception('Angle deg not implemented: {0}'.format(angle_degrees))

        list_object_points.append({
            'camNumber': cam,
            'objectPoints': object_points
        })

    draw_image()

    # Showing image
    cv2.namedWindow('main_window', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('main_window', mouse_callback)
    cv2.imshow('main_window', image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    print('Done!')


def mouse_callback(event, x_image, y_image, flags, param):
    global flag_down, point_init_img, point_end_img

    if event == cv2.EVENT_LBUTTONDOWN:
        print('Event left button down')
        flag_down = True
        point_init_img = [x_image, y_image]
        draw_image()
    elif event == cv2.EVENT_LBUTTONUP:
        print('Event left button up')
        flag_down = False
        point_init_img = [x_image, y_image]
        draw_image()


def draw_image():
    global list_object_points

    image[:, :] = (255, 255, 255)

    for item in list_object_points:
        cam = item['camNumber']
        object_points = item['objectPoints']

        print('Object points for cam {0}: {1}'.format(cam, object_points))
        list_drawn_points = list()
        for point in object_points:
            pos_x = int((point[0] - min_x) * image_width / (max_x - min_x))
            pos_y = int(image_height - (point[1] - min_y) * image_height / (max_y - min_y))

            radius = 3
            cv2.rectangle(image, (pos_x - radius, pos_y - radius), (pos_x + radius, pos_y + radius),
                          (255, 0, 0), -1)
            list_drawn_points.append([pos_x, pos_y])

        print('List drawn points for camera {0}: {1}'.format(cam, list_drawn_points))


if __name__ == '__main__':
    main()
