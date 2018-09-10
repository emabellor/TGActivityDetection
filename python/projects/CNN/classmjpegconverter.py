from classmjpegreader import ClassMjpegReader
from classopenpose import ClassOpenPose
import cv2
import numpy as np
import json
import os.path
import os
from classutils import ClassUtils
from classdescriptors import ClassDescriptors


class ClassMjpegConverter:

    # Method must have extension .mjpeg
    @classmethod
    def convert_video_mjpeg(cls, file_path: str):
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

                    image_np = np.frombuffer(image, dtype="int32")
                    image_cv = cv2.imdecode(image_np, cv2.IMREAD_ANYCOLOR)
                    arr = open_pose.recognize_image(image_cv)  # type: np.ndarray

                    json_dict = {'vectors': arr.tolist()}
                    cls.write_in_file(newFile, ticks, image, json_dict)

                    counter += 1
                    if counter % 10 == 0:
                        print('Counter: ' + str(counter))

            # Deleting mjpegx file if exists and rename
            if os.path.exists(output_video):
                os.remove(output_video)

            # Naming new
            os.rename(file_path_temp, output_video)

    # Method must have extension .mjpegx
    @classmethod
    def convert_video_mjpegx(cls, file_path: str, cam_number: str):
        extension = os.path.splitext(file_path)[1]

        min_percent = 0.05

        if ClassUtils.cam_calib_exists(cam_number):
            calib_params = ClassUtils.load_cam_calib_params(cam_number)
        else:
            print('Warning: Calib params not found for camera {0}'.format(cam_number))
            calib_params = None

        if extension != '.mjpegx':
            raise Exception('file_path must have extension .mjpegx')
        else:
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
                        image_np = np.frombuffer(image, dtype="int32")
                        image_cv = cv2.imdecode(image_np, cv2.IMREAD_ANYCOLOR)

                    params = list()

                    for vector in vectors:
                        # Check if vector is bodypart 25
                        # Avoid confusions with COCO processing
                        if len(vector) != 25:
                            raise Exception('Vector is not bodypart 25 - File: {0} Len Vectors: {1}'.format(
                                file_path, len(vector)
                            ))

                        params.append(ClassDescriptors.get_person_descriptors(vector, min_percent,
                                                                              cam_number=cam_number,
                                                                              image=image_cv,
                                                                              calib_params=calib_params,
                                                                              decode_img=False))
                    # Save object globally
                    # CamNumber
                    new_json_dict = {
                        'vectors': vectors,
                        'params': params,
                        'camNumber': cam_number
                    }

                    # Write in new file - New json dict
                    cls.write_in_file(newFile, ticks, image, new_json_dict)

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
    def save_video_from_list_frames(cls, file_path: str, list_frames):
        extension = os.path.splitext(file_path)[1]

        if extension != '.mjpegx':
            raise Exception('file_path must have extension .mjpegx')
        else:
            output_video = file_path + '_1'

            with open(output_video, 'wb') as newFile:
                counter = 0
                for elem in list_frames:
                    # Avoid previous conversion -> only reading
                    image = elem[0]
                    ticks = elem[1]
                    json_dict = elem[2]

                    # Write in new file
                    cls.write_in_file(newFile, ticks, image, json_dict)

            # Delete old
            os.remove(file_path)

            # Naming new
            os.rename(output_video, file_path)

    @classmethod
    def write_in_file(cls, file, ticks, image, json_dict=None):
        bytes_to_write = cls.get_bytes_file(ticks, image, json_dict)
        file.write(bytes_to_write)

    @staticmethod
    def get_bytes_file(ticks, image, json_dict=None):
        len_json = 0
        len_json_bin = bytes()
        json_bytes = bytes()  # Len 0

        # Support mjpeg and mjpegx conversion
        if json_dict is not None:
            json_string = json.dumps(json_dict)
            json_bytes = bytes(json_string, encoding='utf-8')
            len_json = len(json_bytes)
            len_json_bin = len_json.to_bytes(length=4, byteorder='little')

        len_total = len(image) + len_json + 4  # 4 -> length of the device
        len_total_bin = len_total.to_bytes(length=4, byteorder='little')

        ticks_bin = ticks.to_bytes(length=8, byteorder='little')

        result = len_total_bin + ticks_bin + image

        # Write mjpegx part
        if json_dict is not None:
            result += json_bytes + len_json_bin

        return result