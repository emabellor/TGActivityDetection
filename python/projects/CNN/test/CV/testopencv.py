# Testing classes
import cv2
import numpy as np
from classimage import ClassImage


def main():
    print('Testing opencv functions')

    # Loading image in gray scale
    path = '/home/mauricio/Pictures/429.jpg'
    image = ClassImage.load_gray(path)

    print('Printing image type')
    image_type = str(type(image))
    print(image_type)

    print('Printing size')
    print(image.shape)

    print('Resizing and saving in image')
    shape = (32, 24)
    image_res = ClassImage.resize(image, shape)  # type:np.ndarray

    print('Printing shape 0')
    print(shape[0])

    print('Printing shape 1')
    print(shape[1])

    print(image_res.shape)
    print(shape)

    # Function to overwrite existing elements
    height = image_res.shape[0]
    width = image_res.shape[1]
    with open('/home/mauricio/Pictures/test.txt', 'w') as img_file:
        # Accessing numpy array by elements
        for w in range(width):
            str_image = ''
            for h in range(height):
                if str_image != '':
                    str_image += ','

                str_image += str(image_res[h, w])

            img_file.write(str_image + '\n')

    # File generated
    # Dir File /home/mauricio/Pictures/test.txt
    print('Saving numpy array')
    path_numpy = '/home/mauricio/Datasets/example.npy'
    np.save(path_numpy, image_res)

    print('Done!')
    print('Saving testing opencv')


if __name__ == '__main__':
    main()
