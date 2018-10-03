import os
import numpy as np
import cv2
from classopenpose import ClassOpenPose
from classutils import ClassUtils


def main():
    print('Generating angle descriptors')

    # Initializing instances
    instance_pose = ClassOpenPose()

    # Reading pose from dir path
    image = '/home/mauricio/Pictures/walk.jpg'

    if not os.path.exists(image):
        print('The path {0} does not exists'.format(image))
    else:
        img_cv = cv2.imread(image)

        # Forwarding image
        arr, output_img = instance_pose.recognize_image_tuple(img_cv)
        person_array = arr[0]

        # Generating other descriptors
        # Assume image is OK
        shoulder_dis = ClassUtils.get_euclidean_distance_pt(person_array[1], person_array[2]) + \
                       ClassUtils.get_euclidean_distance_pt(person_array[1], person_array[5])

        torso_dis = ClassUtils.get_euclidean_distance_pt(person_array[1], person_array[8])

        # In total, we have a vector with 8 angles
        # We need to extract the characteristics of the 8 angles

        relation = shoulder_dis / torso_dis
        print(relation)

        cv2.namedWindow('main_window', cv2.WND_PROP_AUTOSIZE)
        cv2.imshow('main_window', output_img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        print('Done!')


if __name__ == '__main__':
    main()
