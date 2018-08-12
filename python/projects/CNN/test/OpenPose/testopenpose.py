import openpose as op
import cv2
import time


def main():
    print('Init testing open pose')
    params = dict()
    params['logging_level'] = 3
    params['output_resolution'] = '-1x-1'
    params['net_resolution'] = '-1x240'
    params['model_pose'] = 'COCO'
    params['alpha_pose'] = 0.6
    params['scale_gap'] = 0.3
    params['scale_number'] = 1
    params['render_threshold'] = 0.05
    params['num_gpu_start'] = 0
    params['disable_blending'] = False
    params['default_model_folder'] = '/home/mauricio/Programs/openpose/openpose/models/'

    openpose = op.OpenPose(params)

    print('Loading image')
    img = cv2.imread('/home/mauricio/Pictures/person.jpg')

    print('Forwarding network')

    start = time.time()
    arr, output_image = openpose.forward(img, True)
    end = time.time()
    print(end - start)

    print('Printing elements')
    print(arr)

    cv2.imshow('output', output_image)
    key = cv2.waitKey(5000)
    cv2.destroyWindow('output')
    print('Done!')


if __name__ == '__main__':
    main()
