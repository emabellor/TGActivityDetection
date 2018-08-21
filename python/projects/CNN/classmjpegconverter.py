from classmjpegreader import ClassMjpegReader
from classopenpose import ClassOpenPose
import cv2
import numpy as np
import json
import os.path
import os
from classutils import ClassUtils


class ClassMjpegConverter:

    # Method must have extension .mjpeg
    @staticmethod
    def convert_video_mjpeg(file_path: str):
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
                    ClassMjpegConverter._write_in_file(newFile, ticks, image, json_dict)

                    counter += 1
                    if counter % 10 == 0:
                        print('Counter: ' + str(counter))

            # Deleting mjpegx file if exists and rename
            if os.path.exists(output_video):
                os.remove(output_video)

            # Naming new
            os.rename(file_path_temp, output_video)

    # Method must have extension .mjpegx
    @ staticmethod
    def convert_video_mjpegx(file_path: str, cam_number: str):
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

                    # Make all processing here
                    positions = ClassMjpegConverter._process_positions(vectors, min_percent, calib_params)
                    json_dict['positions'] = positions
                    json_dict['camNumber'] = cam_number

                    # Write in new file
                    ClassMjpegConverter._write_in_file(newFile, ticks, image, json_dict)

                    counter += 1
                    if counter % 10 == 0:
                        print('Counter: ' + str(counter))
                        if calib_params is None:
                            print('Warning: Calib params not found for camera {0}'.format(cam_number))

            # Delete old
            os.remove(file_path)

            # Naming new
            os.rename(output_video, file_path)

    @staticmethod
    def save_video_from_list_frames(file_path: str, list_frames):
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
                    ClassMjpegConverter._write_in_file(newFile, ticks, image, json_dict)

            # Delete old
            os.remove(file_path)

            # Naming new
            os.rename(output_video, file_path)

    @staticmethod
    def _process_positions(vectors, min_percent, calib_params):
        """
        Calculate position based on the torse
        Project position to the floor in the leg
        """
        positions = []

        for vector in vectors:

            result = ClassUtils.check_vector_integrity_part(vector, min_percent)
            score = 0
            pos_x = 0
            pos_y = 0

            if result and calib_params is not None:
                score = 1

                # Get center of mass of the 3 points
                c_x = (vector[1][0] + vector[8][0] + vector[11][0]) / 3
                c_y = (vector[1][1] + vector[8][1] + vector[11][1]) / 3

                # Project the position to one of the legs
                y_pos = vector[10][1]

                if vector[10][2] < min_percent:
                    y_pos = vector[13][1]

                # Project points
                center = np.array(calib_params['centerPoints'])
                angle_deg = calib_params['angleDegrees']
                homo_mat = np.array(calib_params['homographyMat'])
                projected = ClassUtils.project_points_angle(homo_mat, np.asanyarray([c_x, y_pos], dtype=np.float),
                                                            center, angle_deg)

                # Update points
                pos_x = projected[0]
                pos_y = projected[1]

            positions.append([pos_x, pos_y, score])

        return positions

    @ staticmethod
    def _write_in_file(file, ticks, image, json_dict):
        json_string = json.dumps(json_dict)
        json_string_bin = str.encode(json_string)
        len_json = len(json_string)
        len_json_bin = len_json.to_bytes(length=4, byteorder='little')
        ticks_bin = ticks.to_bytes(length=8, byteorder='little')

        len_total = len(image) + len_json + 4  # 4 -> length of the device
        len_total_bin = len_total.to_bytes(length=4, byteorder='little')

        file.write(len_total_bin)
        file.write(ticks_bin)
        file.write(image)
        file.write(json_string_bin)
        file.write(len_json_bin)
