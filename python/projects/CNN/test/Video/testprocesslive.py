import cv2
import numpy as np
from classopenpose import ClassOpenPose
from classutils import ClassUtils
from classnn import ClassNN
from classdescriptors import ClassDescriptors


def main():
    print('Initializing main function')

    # Prompt for user input
    cam_number_str = input('Insert camera number to process: ')
    cam_number = int(cam_number_str)

    # Open video from opencv
    cap = cv2.VideoCapture(cam_number)

    # Initializing open pose distance
    instance_pose = ClassOpenPose()

    # Initializing variables
    model_dir = '/home/mauricio/models/nn_classifier'
    instance_nn = ClassNN.load_from_params(model_dir)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Processing frame with openpose
        arr, frame = instance_pose.recognize_image_tuple(frame)

        # Check if there is one frame with vector integrity
        arr_pass = list()

        min_score = 0.05
        # Checking vector integrity for all elements
        # Verify there is at least one arm and one leg
        for elem in arr:
            if ClassUtils.check_vector_integrity_part(elem, min_score):
                arr_pass.append(elem)

        if len(arr_pass) != 1:
            print('Invalid len for arr_pass: {0}'.format(arr_pass))
        else:
            person_array = arr_pass[0]

            # Getting person descriptors
            results = ClassDescriptors.get_person_descriptors(person_array, min_score)

            # Descriptors
            data_to_add = results['angles']
            data_to_add += ClassUtils.get_flat_list(results['transformed_points'])

            data_np = np.asanyarray(data_to_add, dtype=np.float)

            # Getting result predict
            result_predict = instance_nn.predict_model(data_np)
            detected_class = result_predict['classes']

            label_class = get_label_name(instance_nn.label_names, detected_class)
            print('Detected: {0} - Label: {1}'.format(detected_class, label_class))

            # Draw text class into image - Evaluation purposes
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (255, 255, 255)
            line_type = 2

            cv2.putText(frame, '{0}'.format(label_class),
                        (0, 0),
                        font,
                        font_scale,
                        font_color,
                        line_type)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    print('Done!')


def get_label_name(label_names, label):
    for label_item in label_names:
        label_name = label_item[0]
        label_id = label_item[1]

        if label_id == label:
            return label_name

    raise Exception('Cant find label with id {0}'.format(label))


if __name__ == '__main__':
    main()
