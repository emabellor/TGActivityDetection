from classmjpegreader import ClassMjpegReader
from classopenpose import ClassOpenPose
import cv2
import numpy as np
import json
import os.path
import os
from classutils import ClassUtils
from classnn import ClassNN
from classdescriptors import ClassDescriptors


class ClassMjpegConverter:
    min_percent = 0.05

    # Method must have extension .mjpeg
    @classmethod
    def convert_video_mjpeg(cls, file_path: str, delete_mjpeg=False):
        extension = os.path.splitext(file_path)[1]

        if extension != '.mjpeg':
            raise Exception('file_path must have extension .mjpeg')
        else:
            output_video = file_path.replace('.mjpeg', '.mjpegx')
            print('output_video: ' + output_video)

            # Converting first with _1 to avoid confusions
            file_path_temp = output_video + '_1'

            print('Initializing instance')
            open_pose = ClassOpenPose()

            images = ClassMjpegReader.process_video(file_path)
            print('Len images: ' + str(len(images)))
            with open(file_path_temp, 'wb') as newFile:
                counter = 0
                for elem in images:
                    image = elem[0]
                    ticks = elem[1]

                    image_np = np.frombuffer(image, dtype=np.uint8)
                    image_cv = cv2.imdecode(image_np, cv2.IMREAD_ANYCOLOR)
                    arr = open_pose.recognize_image(image_cv)  # type: np.ndarray

                    json_dict = {'vectors': arr.tolist()}
                    ClassUtils.write_in_file(newFile, ticks, image, json_dict)

                    counter += 1
                    if counter % 10 == 0:
                        print('Counter: ' + str(counter))

            # Deleting mjpegx file if exists and rename
            if os.path.exists(output_video):
                os.remove(output_video)

            # Naming new
            os.rename(file_path_temp, output_video)

            if delete_mjpeg:
                os.remove(file_path)

    # Method must have extension .mjpegx
    @classmethod
    def convert_video_mjpegx(cls, file_path: str, cam_number: str, instance_nn_pose: ClassNN = None):
        extension = os.path.splitext(file_path)[1]

        if ClassUtils.cam_calib_exists(cam_number):
            calib_params = ClassUtils.load_cam_calib_params(cam_number)
        else:
            print('Warning: Calib params not found for camera {0}'.format(cam_number))
            calib_params = None

        if extension != '.mjpegx':
            raise Exception('file_path must have extension .mjpegx')

        print('Reading image')
        output_video = file_path + '_1'

        images = ClassMjpegReader.process_video_mjpegx(file_path)
        print('Len images: ' + str(len(images)))

        with open(output_video, 'wb') as newFile:
            counter = 0
            for elem in images:
                # Avoid previous conversion -> only reading
                image = elem[0]
                ticks = elem[1]
                json_dict = elem[2]

                vectors = json_dict['vectors']

                # Avoid over processing - Only process images with poses
                image_cv = None
                if len(vectors) != 0:
                    image_np = np.frombuffer(image, dtype=np.uint8)
                    image_cv = cv2.imdecode(image_np, cv2.IMREAD_ANYCOLOR)

                params = list()

                for vector in vectors:
                    # Check if vector is bodypart 25
                    # Avoid confusions with COCO processing
                    if len(vector) != 25:
                        raise Exception('Vector is not bodypart 25 - File: {0} Len Vectors: {1}'.format(
                            file_path, len(vector)
                        ))

                    params.append(ClassDescriptors.get_person_descriptors(vector, cls.min_percent,
                                                                          cam_number=cam_number,
                                                                          image=image_cv,
                                                                          calib_params=calib_params,
                                                                          decode_img=False,
                                                                          instance_nn_pose=instance_nn_pose))

                # Save object globally
                # CamNumber
                new_json_dict = {
                    'vectors': vectors,
                    'params': params,
                    'camNumber': cam_number
                }

                # Write in new file - New json dict
                ClassUtils.write_in_file(newFile, ticks, image, new_json_dict)

                counter += 1
                if counter % 100 == 0:
                    print('Counter: ' + str(counter))
                    if calib_params is None:
                        print('Warning: Calib params not found for camera {0}'.format(cam_number))

        # Delete old
        os.remove(file_path)

        # Naming new
        os.rename(output_video, file_path)

    @classmethod
    def convert_video_mjpegx_reid(cls, file_path: str, list_people_reid=None, save_img=False, option=None):
        extension = os.path.splitext(file_path)[1]

        if extension != '.mjpegx':
            raise Exception('file_path must have extension .mjpegx')

        print('Reading image')
        output_video = file_path + '_1'

        images = ClassMjpegReader.process_video_mjpegx(file_path)
        print('Len images: ' + str(len(images)))

        with open(output_video, 'wb') as newFile:
            counter = 0
            for elem in images:
                # Avoid previous conversion -> only reading
                image = elem[0]
                ticks = elem[1]
                json_dict = elem[2]

                params = json_dict['params']

                # Update elements
                for param in params:
                    found = False
                    for person_reid in list_people_reid:
                        for item in person_reid.list_poses:
                            if item['poseGuid'] == param['poseGuid']:
                                param['personGuid'] = person_reid.person_guid
                                found = True
                                break

                        if found:
                            # Saving image with pose identified
                            if save_img:
                                cnn_folder = ClassUtils.cnn_folder
                                if option is not None:
                                    cnn_folder = os.path.join(cnn_folder, option)

                                path_img = os.path.join(cnn_folder, param['personGuid'])
                                if not os.path.exists(path_img):
                                    os.makedirs(path_img)

                                filename = os.path.join(path_img, '{0}.jpg'.format(ticks))
                                image_arr = np.frombuffer(image, dtype=np.uint8)
                                image_cv = cv2.imdecode(image_arr, cv2.IMREAD_ANYCOLOR)

                                ClassDescriptors.draw_pose(image_cv, param['vectors'], cls.min_percent,
                                                           param['keyPose'])
                                cv2.imwrite(filename, image_cv)

                            break

                    if not found:
                        param['personGuid'] = ''

                # Saving data in json dict
                ClassUtils.write_in_file(newFile, ticks, image, json_dict)

                counter += 1
                if counter % 100 == 0:
                    print('Counter: ' + str(counter))

        # Delete old
        os.remove(file_path)

        # Naming new
        os.rename(output_video, file_path)

    @classmethod
    def save_video_from_list_frames(cls, file_path: str, list_frames):
        extension = os.path.splitext(file_path)[1]

        if extension != '.mjpegx':
            raise Exception('file_path must have extension .mjpegx')

        output_video = file_path + '_1'

        with open(output_video, 'wb') as newFile:
            counter = 0
            for elem in list_frames:
                # Avoid previous conversion -> only reading
                image = elem[0]
                ticks = elem[1]
                json_dict = elem[2]

                # Write in new file
                ClassUtils.write_in_file(newFile, ticks, image, json_dict)

        # Delete old
        os.remove(file_path)

        # Naming new
        os.rename(output_video, file_path)
