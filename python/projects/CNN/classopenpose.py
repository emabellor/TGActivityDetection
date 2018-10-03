import numpy as np
import openpose as op
import cv2
from sys import platform


class ClassOpenPose:

    def __init__(self):
        print('Initializing constructor')

        params = dict()
        params['logging_level'] = 3
        params['output_resolution'] = "-1x-1"
        params['net_resolution'] = "-1x240"
        # params['model_pose'] = "COCO"
        # params['model_pose'] = "MPI"
        params['model_pose'] = "BODY_25"
        params['alpha_pose'] = 0.6
        params['scale_gap'] = 0.3
        params['scale_number'] = 1
        params['render_threshold'] = 0.05
        params['num_gpu_start'] = 0
        params['disable_blending'] = False
        params['default_model_folder'] = "/home/mauricio/Programs/openpose/openpose/models/"

        # Static region
        if platform == 'win32':
            params['default_model_folder'] = 'C:\\OpenPose\\openpose-master\\models\\'

        self.open_pose = op.OpenPose(params)
        print('Openpose initialized')

    def recognize_image(self, image: np.ndarray):
        arr, output_image = self.open_pose.forward(image, True)
        return arr

    def recognize_image_draw(self, image: np.ndarray):
        print('Recognizing image')
        arr, output_image = self.open_pose.forward(image, True)
        return output_image

    def recognize_image_tuple(self, image: np.ndarray):
        print('Recognizing image')
        arr, output_image = self.open_pose.forward(image, True)
        return arr, output_image
