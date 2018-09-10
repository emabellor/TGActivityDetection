from classutils import ClassUtils
import math
import numpy as np
import cv2
from tkinter.filedialog import askopenfilename
from classopenpose import ClassOpenPose
import os
import json
import time


# Static class
class ClassDescriptors:

    @classmethod
    def get_person_descriptors(cls, person_array, min_pose_score, cam_number=0, image=None, calib_params=None,
                               decode_img=False):

        # Person descriptors for vector integrity
        # In function ClassUtils.check_vector_integrity_part
        # One part of the arms and one of the legs must exist
        # Vectors 11 and 14 are optional
        # Vectors of the torso must exist
        if not decode_img:
            image_np = image
        else:
            image_buffer = np.frombuffer(image, dtype="int32")
            image_np = cv2.imdecode(image_buffer, cv2.IMREAD_ANYCOLOR)

        if ClassUtils.check_vector_integrity_pos(person_array, min_pose_score):
            integrity = True
            only_pos = ClassUtils.check_vector_only_pos(person_array, min_pose_score)
        else:
            integrity = False
            only_pos = False

        # Getting relation torso
        relation = ClassDescriptors._get_torso_shoulders_relation(person_array, min_pose_score,
                                                                  only_pos=only_pos, integrity=integrity)

        # Getting angles
        list_angles, list_angles_degrees = ClassDescriptors._get_person_descriptors_angles(person_array, min_pose_score,
                                                                                           only_pos=only_pos,
                                                                                           integrity=integrity)

        # Getting transformed points
        transformed_points = ClassDescriptors._get_transformed_points(person_array, min_pose_score,
                                                                      only_pos=only_pos, integrity=integrity)

        # Getting descriptor for neural net
        full_desc = list_angles
        full_desc += ClassUtils.get_flat_list(transformed_points)

        info_upper = cls._process_upper_color(person_array, min_pose_score, image_np, integrity=integrity)
        info_lower = cls._process_lower_color(person_array, min_pose_score, image_np, integrity=integrity)

        # Getting hist - BTF transformation
        hists = cls.get_color_histograms(person_array, min_pose_score, image_np, decode_img=False,
                                         cumulative=False, normalize=False, integrity=integrity)

        local_position, global_position = cls._process_position_vector(person_array, min_pose_score, calib_params)

        result = {
            'vectors': person_array,
            'relation': relation,
            'angles': list_angles,
            'anglesDegrees': list_angles_degrees,
            'transformedPoints': transformed_points,
            'fullDesc': full_desc,
            'onlyPos': only_pos,
            'integrity': integrity,
            'colorUpper': info_upper[1],
            'colorLower': info_lower[1],
            'hists': hists,
            'camNumber': cam_number,
            'localPosition': local_position,
            'globalPosition': global_position
        }

        return result

    @staticmethod
    def _process_position_vector(vector, min_percent, calib_params):
        """
        Calculate position based on the torse
        Project position to the floor in the leg
        """

        result = ClassUtils.check_vector_integrity_pos(vector, min_percent)
        score = 0
        global_pos_x = 0
        global_pos_y = 0
        local_pos_x = 0
        local_pos_y = 0

        if result and calib_params is not None:
            score = 1

            vector_guess = ClassUtils.complete_points(vector, min_percent)

            local_pos_x = vector_guess[8][0]

            # Project the position to the max point of the leg
            local_pos_y = vector_guess[11][1]

            if vector_guess[14][1] > local_pos_y:
                local_pos_y = vector_guess[14][1]

            # Project points
            center = np.array(calib_params['centerPoints'])
            angle_deg = calib_params['angleDegrees']
            homo_mat = np.array(calib_params['homographyMat'])
            projected = ClassUtils.project_points_angle(homo_mat, np.asanyarray([local_pos_x, local_pos_y],
                                                        dtype=np.float), center, angle_deg)

            # Update points
            global_pos_x = projected[0]
            global_pos_y = projected[1]

        local_position = [local_pos_x, local_pos_y, score]
        global_position = [global_pos_x, global_pos_y, score]

        return local_position, global_position

    @classmethod
    # Histograms are cumulative
    def get_color_histograms(cls, vector: list, min_percent: float, image, decode_img=True, cumulative=True,
                             normalize=True, integrity=True):

        red_hist = [0 for _ in range(256)]
        green_hist = [0 for _ in range(256)]
        blue_hist = [0 for _ in range(256)]
        len_items = 0

        if not integrity:
            # Dummy command
            # Does not do anything
            len_items = 0
        else:
            # Checking vector integrity
            if not ClassUtils.check_vector_integrity_pos(vector, min_percent):
                raise Exception('Vector integrity not valid')

            if not decode_img:
                image_np = image
            else:
                image_buffer = np.frombuffer(image, dtype="int32")
                image_np = cv2.imdecode(image_buffer, cv2.IMREAD_ANYCOLOR)

            upper_items = cls._process_upper_color(vector, min_percent, image_np)[0]
            lower_items = cls._process_lower_color(vector, min_percent, image_np)[0]

            for item in upper_items:
                red_hist[item[0]] += 1
                green_hist[item[1]] += 1
                blue_hist[item[2]] += 1
            len_items += len(upper_items)

            for item in lower_items:
                red_hist[item[0]] += 1
                green_hist[item[1]] += 1
                blue_hist[item[2]] += 1
            len_items += len(lower_items)

            # Normalize histogram
            if normalize:
                red_hist = [x / len_items for x in red_hist]
                green_hist = [x / len_items for x in green_hist]
                blue_hist = [x / len_items for x in blue_hist]

            # Get cumulative hist
            if cumulative:
                red_hist_cum = [0 for _ in range(256)]
                green_hist_cum = [0 for _ in range(256)]
                blue_hist_cum = [0 for _ in range(256)]

                for index in range(len(red_hist)):
                    red_hist_cum[index] = sum(red_hist[:index+1])
                    green_hist_cum[index] = sum(green_hist[:index+1])
                    blue_hist_cum[index] = sum(blue_hist[:index+1])

                red_hist = red_hist_cum
                green_hist = green_hist_cum
                blue_hist = blue_hist_cum

        return red_hist, green_hist, blue_hist

    @classmethod
    def get_cumulative_hists(cls, hists):
        len_red = 0
        len_green = 0
        len_blue = 0

        for i in range(256):
            len_red += hists[0][i]
            len_green += hists[1][i]
            len_blue += hists[2][i]

        if len_red != len_green or len_green != len_blue or len_red != len_blue:
            raise Exception('Illegal len for elems {0} {1} {2}'.format(len_red, len_green, len_blue))

        len_items = len_red

        # Normalize histogram
        red_hist = [x / len_items for x in hists[0]]
        green_hist = [x / len_items for x in hists[1]]
        blue_hist = [x / len_items for x in hists[2]]

        # Get cumulative hist
        red_hist_cum = [0 for _ in range(256)]
        green_hist_cum = [0 for _ in range(256)]
        blue_hist_cum = [0 for _ in range(256)]

        for index in range(len(red_hist)):
            red_hist_cum[index] = sum(red_hist[:index + 1])
            green_hist_cum[index] = sum(green_hist[:index + 1])
            blue_hist_cum[index] = sum(blue_hist[:index + 1])

        red_hist = red_hist_cum
        green_hist = green_hist_cum
        blue_hist = blue_hist_cum

        return red_hist, green_hist, blue_hist

    @classmethod
    def transform_image(cls, image: np.ndarray, hist_ori: list, hist_dst: list):
        # Get color table for each element
        # Lookup table
        ct_red = [0 for _ in range(256)]
        ct_green = [0 for _ in range(256)]
        ct_blue = [0 for _ in range(256)]

        last_val_red = 0
        last_val_green = 0
        last_val_blue = 0
        for index in range(256):
            ct_red[index] = cls.get_value_ct_btf(index, hist_ori[0], hist_dst[0], last_val_red)
            ct_green[index] = cls.get_value_ct_btf(index, hist_ori[1], hist_dst[1], last_val_green)
            ct_blue[index] = cls.get_value_ct_btf(index, hist_ori[2], hist_dst[2], last_val_blue)

            last_val_red = ct_red[index]
            last_val_green = ct_green[index]
            last_val_blue = ct_blue[index]

        # Copy image and transform depending on lookup table
        image_res = np.copy(image)

        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                # Images are in BGR format
                # Remember!
                image_res[row, col, 0] = ct_blue[image[row, col, 0]]
                image_res[row, col, 1] = ct_green[image[row, col, 1]]
                image_res[row, col, 2] = ct_red[image[row, col, 2]]

        # Done - Returning image_res
        return image_res

    @classmethod
    def get_value_ct_btf(cls, value, hist_ori, hist_dst, last_val):
        allow_delta = 1 / 256

        # Based in orazio et al
        hst_value_ori = hist_ori[value]
        hst_value_dst = hist_dst[value]

        if math.fabs(hst_value_ori - hst_value_dst) < allow_delta and value >= last_val:
            # Return value if delta is low
            return value

        # Start incrementing until there is a match
        value_dst = 0
        for x in range(256):
            value_dst = x
            if hst_value_ori <= hist_dst[x] and value_dst >= last_val:
                # There is a match
                break

        return value_dst

    @classmethod
    def process_colors(cls, vectors: list, min_percent: float, image, decode_img=True):
        arr_pass = list()
        list_colors_upper = list()
        list_colors_lower = list()

        for vector in vectors:
            if ClassUtils.check_vector_integrity_pos(vector, min_percent):
                arr_pass.append(vector)

        if len(arr_pass) == 0:
            # Only decode images which have valid people
            for _ in vectors:
                list_colors_upper.append(0)
                list_colors_lower.append(0)
        else:
            if not decode_img:
                image_np = image
            else:
                image_buffer = np.frombuffer(image, dtype="int32")
                image_np = cv2.imdecode(image_buffer, cv2.IMREAD_ANYCOLOR)

            for vector in vectors:
                if not ClassUtils.check_vector_integrity_pos(vector, min_percent):
                    list_colors_upper.append(0)
                    list_colors_lower.append(0)
                else:
                    upper_color = cls._process_upper_color(vector, min_percent, image_np)[1]
                    lower_color = cls._process_lower_color(vector, min_percent, image_np)[1]

                    list_colors_upper.append(upper_color)
                    list_colors_lower.append(lower_color)

        # Assert sentence
        if len(vectors) != len(list_colors_upper):
            raise Exception('Error - Len vectors are different!')

        return list_colors_upper, list_colors_lower

    @classmethod
    def process_colors_person(cls, person_vector, min_percent, image, decode_img=True):
        # Assume check_vector_integrity checked
        if not decode_img:
            image_np = image
        else:
            image_buffer = np.frombuffer(image, dtype="int32")
            image_np = cv2.imdecode(image_buffer, cv2.IMREAD_ANYCOLOR)

        upper_color = cls._process_upper_color(person_vector, min_percent, image_np)[1]
        lower_color = cls._process_lower_color(person_vector, min_percent, image_np)[1]

        return upper_color, lower_color

    @classmethod
    def _process_upper_color(cls, vector: list, min_percent: float, image_np, integrity=True):
        # Process color between shoulders and middle of the skeleton
        color_items = list()
        if not integrity or image_np is None:
            mean_colors = [0, 0, 0]
        else:
            # Process color between shoulders and middle of the skeleton
            color_items = list()

            color_items += cls._get_color_items(vector[1], vector[8], image_np)

            # Check left shoulder
            if ClassUtils.check_point_integrity(vector[2], min_percent):
                color_items += cls._get_color_items(vector[1], vector[2], image_np)

            # Check right shoulder
            if ClassUtils.check_point_integrity(vector[5], min_percent):
                color_items += cls._get_color_items(vector[1], vector[5], image_np)

            mean_colors = cls._get_mean_colors(color_items)

        return color_items, mean_colors

    @classmethod
    def _process_lower_color(cls, vector: list, min_percent: float, image_np, integrity=True):
        # Process colors in lower part of the skeleton
        color_items = list()

        if not integrity or image_np is None:
            mean_colors = [0, 0, 0]
        else:
            if ClassUtils.check_point_integrity(vector[9], min_percent) \
                    and ClassUtils.check_point_integrity(vector[10], min_percent):
                color_items += cls._get_color_items(vector[9], vector[10], image_np)

            if ClassUtils.check_point_integrity(vector[10], min_percent) \
                    and ClassUtils.check_point_integrity(vector[11], min_percent):
                color_items += cls._get_color_items(vector[10], vector[11], image_np)

            if ClassUtils.check_point_integrity(vector[12], min_percent) \
                    and ClassUtils.check_point_integrity(vector[13], min_percent):
                color_items += cls._get_color_items(vector[12], vector[13], image_np)

            if ClassUtils.check_point_integrity(vector[13], min_percent) \
                    and ClassUtils.check_point_integrity(vector[14], min_percent):
                color_items += cls._get_color_items(vector[13], vector[14], image_np)

            mean_colors = cls._get_mean_colors(color_items)

        return color_items, mean_colors

    @classmethod
    def _get_mean_colors(cls, color_items):
        # Get color mean
        # Assume color are in rgb color space
        # BGR conversion performed with get_color_items function
        color_items_np = np.array(color_items)
        list_r = color_items_np[:, 0]
        list_g = color_items_np[:, 1]
        list_b = color_items_np[:, 2]

        # Compute value by 2
        mean_r = ClassUtils.compute_mean(list_r)
        mean_g = ClassUtils.compute_mean(list_g)
        mean_b = ClassUtils.compute_mean(list_b)

        # Return to normal value
        return [mean_r, mean_g, mean_b]

    @classmethod
    def _get_color_items(cls, point1, point2, image_np):
        # Check delta x or delta y
        # Avoid big slope and maximize evaluation points
        delta_x = int(math.fabs(point1[0] - point2[0]))
        delta_y = int(math.fabs(point1[1] - point2[1]))

        if delta_x == 0 and delta_y == 0:
            # Return empty list
            return list()
        elif delta_x > delta_y:
            return cls._get_color_items_x(point1, point2, image_np)
        else:
            return cls._get_color_items_y(point1, point2, image_np)

    @staticmethod
    def _get_color_items_x(point1, point2, image_np):
        color_items = list()

        # Processing color mean into list
        # Get rect equation
        # y = mx + b
        m = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = (point1[1] - m * point1[0])

        # Get xmin and xmax into list
        x_min = int(point1[0])
        x_max = int(point2[0])

        if x_min > x_max:
            x_min = int(point2[0])
            x_max = int(point1[0])

        # Generating point into range
        for x in range(x_min, x_max):
            # Apply rect equation
            y = int((m * x) + b)

            # Getting pixel
            pixel = image_np[y, x]

            # Adding color item
            # Change bgr color space
            color_items.append([pixel[2], pixel[1], pixel[0]])

        return color_items

    @staticmethod
    def _get_color_items_y(point1, point2, image_np):
        color_items = list()

        # Processing color mean into list
        # Get rect equation
        # x = my + b
        m = (point2[0] - point1[0]) / (point2[1] - point1[1])
        b = (point1[0] - m * point1[1])

        # Get ymin and ymax into list
        y_min = int(point1[1])
        y_max = int(point2[1])

        if y_min > y_max:
            y_min = int(point2[1])
            y_max = int(point1[1])

        # Generating point into range
        for y in range(y_min, y_max):
            # Apply rect equation
            x = int((m * y) + b)

            # Getting pixel
            pixel = image_np[y, x]

            # Adding color item
            # Change bgr color space
            color_items.append([pixel[2], pixel[1], pixel[0]])

        return color_items

    @staticmethod
    def _get_person_descriptors_angles(person_array, min_pose_score, only_pos=False, integrity=True):
        # Function compatible with check vector integrity pos
        # In total we have 12 angles
        # and 4 angles related with torso and shoulders
        # If only pos enabled, does not calculate angles in arms

        # Arms angles
        # If both arms are valid iterate from each point
        # If one of the arms is valid iterate and repeat
        list_angles = list()
        list_angles_degrees = list()

        if not integrity:
            for i in range(12):
                list_angles.append(0)
                list_angles_degrees.append(0)
        else:
            # Depends of the integrity of the points listed in list
            points_arm_right = [person_array[2], person_array[3], person_array[4]]
            points_arm_left = [person_array[5], person_array[6], person_array[7]]

            points_leg_right = [person_array[9], person_array[10], person_array[11]]
            points_leg_left = [person_array[12], person_array[13], person_array[14]]
            points_leg_partial = [person_array[9], person_array[10], person_array[12], person_array[13]]

            def add_arm_right():
                list_angles.append(ClassUtils.get_angle(person_array[3], person_array[2], person_array[1], ))
                list_angles.append(ClassUtils.get_angle(person_array[4], person_array[3], person_array[2]))
                list_angles.append(ClassUtils.get_angle_lines(person_array[3], person_array[2],
                                                              person_array[1], person_array[8]))

            def add_arm_left():
                list_angles.append(ClassUtils.get_angle(person_array[1], person_array[5], person_array[6]))
                list_angles.append(ClassUtils.get_angle(person_array[5], person_array[6], person_array[7]))
                list_angles.append(ClassUtils.get_angle_lines(person_array[6], person_array[5],
                                                              person_array[1], person_array[8]))

            if only_pos:
                # Add torso angles as zero
                for _ in range(6):
                    list_angles.append(0)
            elif ClassUtils.check_point_list(points_arm_right, min_pose_score) \
                    and ClassUtils.check_point_list(points_arm_left, min_pose_score):
                # Get angle for points and torso points
                add_arm_right()
                add_arm_left()
            elif ClassUtils.check_point_list(points_arm_right, min_pose_score):
                for _ in range(2):
                    add_arm_right()
            elif ClassUtils.check_point_list(points_arm_left, min_pose_score):
                for _ in range(2):
                    add_arm_left()
            else:
                raise Exception('Cant find valid arms integrity')

            # Leg angles
            # Points 11 and 14 are optional
            # If there are no points, extend arm and get 0
            # If there are no 11 and 14, get angles and assume angle 0
            def add_leg_right():
                list_angles.append(ClassUtils.get_angle(person_array[10], person_array[9], person_array[8]))
                list_angles.append(ClassUtils.get_angle(person_array[11], person_array[10], person_array[9]))
                list_angles.append(ClassUtils.get_angle_lines(person_array[10], person_array[9],
                                                              person_array[1], person_array[8]))

            def add_leg_left():
                list_angles.append(ClassUtils.get_angle(person_array[8], person_array[12], person_array[13]))
                list_angles.append(ClassUtils.get_angle(person_array[12], person_array[13], person_array[14]))
                list_angles.append(ClassUtils.get_angle_lines(person_array[13], person_array[12],
                                                              person_array[1], person_array[8]))

            def add_leg_partial():
                list_angles.append(ClassUtils.get_angle(person_array[10], person_array[9], person_array[8]))
                list_angles.append(math.pi)  # Invalid point - Assume pi angle - 180 degrees
                list_angles.append(ClassUtils.get_angle_lines(person_array[10], person_array[9],
                                                              person_array[1], person_array[8]))

                list_angles.append(ClassUtils.get_angle(person_array[8], person_array[12], person_array[13]))
                list_angles.append(math.pi)  # Invalid point - Assume pi angle - 180 degrees
                list_angles.append(ClassUtils.get_angle_lines(person_array[13], person_array[12],
                                                              person_array[1], person_array[8]))

            if ClassUtils.check_point_list(points_leg_right, min_pose_score) \
                    and ClassUtils.check_point_list(points_leg_left, min_pose_score):
                add_leg_right()
                add_leg_left()
            elif ClassUtils.check_point_list(points_leg_right, min_pose_score):
                for _ in range(2):
                    add_leg_right()
            elif ClassUtils.check_point_list(points_leg_left, min_pose_score):
                for _ in range(2):
                    add_leg_left()
            elif ClassUtils.check_point_list(points_leg_partial, min_pose_score):
                add_leg_partial()
            else:
                raise Exception('Cant find valid legs integrity')

            for angle in list_angles:
                list_angles_degrees.append(angle * 180 / math.pi)

        return list_angles, list_angles_degrees

    @staticmethod
    def _get_torso_shoulders_relation(person_array, min_pose_score, only_pos=False, integrity=True):
        # Function compatible with check_vector integrity pos
        # If there is not shoulder (only pos) return zero
        # Otherwise performs calculation
        # Depends of vector integrity of shoulders and arms

        if only_pos or not integrity:
            relation_shoulders = 0
        else:
            torso_dis = ClassUtils.get_euclidean_point(person_array[1], person_array[8])

            if person_array[2][2] >= min_pose_score and person_array[5][2] >= min_pose_score:
                shoulder_dis = ClassUtils.get_euclidean_point(person_array[1], person_array[2]) + \
                               ClassUtils.get_euclidean_point(person_array[1], person_array[5])
            elif person_array[2][2] >= min_pose_score:
                shoulder_dis = 2 * ClassUtils.get_euclidean_point(person_array[1], person_array[2])
            elif person_array[5][2] >= min_pose_score:
                shoulder_dis = 2 * ClassUtils.get_euclidean_point(person_array[1], person_array[5])
            else:
                raise Exception('Invalid confidence for shoulder points')

            relation_shoulders = shoulder_dis / torso_dis

        return relation_shoulders

    @staticmethod
    def _get_transformed_points(person_array, min_pose_score, only_pos=False, integrity=True):
        # 12 points in list

        list_points = list()
        if not integrity:
            # If not integrity, add points as zero
            for _ in range(12):
                list_points.append([0, 0])
        else:
            # Function compatible with check_vector_integrity_part
            # Depends of vector integrity of shoulders and arms
            neck_point = person_array[1]
            torso_dis = ClassUtils.get_euclidean_point(person_array[1], person_array[8])

            # Let point 1 be the origin
            # Transform the rest of the points

            points_arm_right = [person_array[2], person_array[3], person_array[4]]
            points_arm_left = [person_array[5], person_array[6], person_array[7]]

            points_leg_right = [person_array[9], person_array[10], person_array[11]]
            points_leg_left = [person_array[12], person_array[13], person_array[14]]
            points_leg_partial = [person_array[9], person_array[10], person_array[12], person_array[13]]

            # Arms points
            def add_arm_right():
                list_points.append(ClassDescriptors._transform_point(person_array[2], neck_point, torso_dis))
                list_points.append(ClassDescriptors._transform_point(person_array[3], neck_point, torso_dis))
                list_points.append(ClassDescriptors._transform_point(person_array[4], neck_point, torso_dis))

            def add_arm_left():
                list_points.append(ClassDescriptors._transform_point(person_array[5], neck_point, torso_dis))
                list_points.append(ClassDescriptors._transform_point(person_array[6], neck_point, torso_dis))
                list_points.append(ClassDescriptors._transform_point(person_array[7], neck_point, torso_dis))

            if only_pos:
                # Add six points in blank - only pos element!
                for _ in range(6):
                    list_points.append([0, 0])
            elif ClassUtils.check_point_list(points_arm_right, min_pose_score) \
                    and ClassUtils.check_point_list(points_arm_left, min_pose_score):
                add_arm_right()
                add_arm_left()
            elif ClassUtils.check_point_list(points_arm_right, min_pose_score):
                for _ in range(2):
                    add_arm_right()
            elif ClassUtils.check_point_list(points_arm_left, min_pose_score):
                for _ in range(2):
                    add_arm_left()
            else:
                raise Exception('Cant find valid arms integrity')

            # Torso points
            list_points.append(ClassDescriptors._transform_point(person_array[8], neck_point, torso_dis))

            # Legs points
            # Check add_leg_partial_function
            def add_leg_right():
                list_points.append(ClassDescriptors._transform_point(person_array[9], neck_point, torso_dis))
                list_points.append(ClassDescriptors._transform_point(person_array[10], neck_point, torso_dis))
                list_points.append(ClassDescriptors._transform_point(person_array[11], neck_point, torso_dis))

            def add_leg_left():
                list_points.append(ClassDescriptors._transform_point(person_array[12], neck_point, torso_dis))
                list_points.append(ClassDescriptors._transform_point(person_array[13], neck_point, torso_dis))
                list_points.append(ClassDescriptors._transform_point(person_array[14], neck_point, torso_dis))

            # Points 11 and 14 not valid
            # Try to guess points
            def add_leg_partial():
                list_points.append(ClassDescriptors._transform_point(person_array[9], neck_point, torso_dis))
                list_points.append(ClassDescriptors._transform_point(person_array[10], neck_point, torso_dis))
                point_11 = ClassDescriptors._guess_leg_point(person_array[9], person_array[10])
                list_points.append(ClassDescriptors._transform_point(point_11, neck_point, torso_dis))

                list_points.append(ClassDescriptors._transform_point(person_array[12], neck_point, torso_dis))
                list_points.append(ClassDescriptors._transform_point(person_array[13], neck_point, torso_dis))
                point_14 = ClassDescriptors._guess_leg_point(person_array[12], person_array[13])
                list_points.append(ClassDescriptors._transform_point(point_14, neck_point, torso_dis))

            if ClassUtils.check_point_list(points_leg_right, min_pose_score) \
                    and ClassUtils.check_point_list(points_leg_left, min_pose_score):
                add_leg_right()
                add_leg_left()
            elif ClassUtils.check_point_list(points_leg_right, min_pose_score):
                for _ in range(2):
                    add_leg_right()
            elif ClassUtils.check_point_list(points_leg_left, min_pose_score):
                for _ in range(2):
                    add_leg_left()
            elif ClassUtils.check_point_list(points_leg_partial, min_pose_score):
                add_leg_partial()
            else:
                raise Exception('Cant find valid legs integrity')

        return list_points

    @staticmethod
    def _transform_point(point, base_point, base_len):
        # Not add confidence
        # Unroll when training
        new_x = (point[0] - base_point[0]) / base_len
        new_y = (point[1] - base_point[1]) / base_len

        new_point = [new_x, new_y]
        return new_point

    @staticmethod
    def _guess_leg_point(point1, point2):
        # point 1 must be the base point
        # point 2 must be the extended point

        # Not add confidence
        # Unroll when training

        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]

        new_x = point2[0] + delta_x
        new_y = point2[1] + delta_y

        point = [new_x, new_y]
        return point

    @classmethod
    def get_width_relation(cls, person_array: list):
        # Vector integrity must be checked first
        # Watch this post
        # https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
        torso_dis = ClassUtils.get_euclidean_distance_pt(person_array[1], person_array[8])

        width = int(torso_dis / 15) + 1
        return width

    @classmethod
    def get_points_by_pose(cls, image: np.ndarray, pose, min_score, draw=False):
        # Pose must be checked with calc pose first
        # List points is in rgb format
        list_points = list()
        cls.add_list_pt(list_points, pose[1], pose[8], min_score)

        cls.add_list_pt(list_points, pose[1], pose[2], min_score)
        cls.add_list_pt(list_points, pose[2], pose[3], min_score)
        cls.add_list_pt(list_points, pose[3], pose[4], min_score)

        cls.add_list_pt(list_points, pose[1], pose[5], min_score)
        cls.add_list_pt(list_points, pose[5], pose[6], min_score)
        cls.add_list_pt(list_points, pose[6], pose[7], min_score)

        return_pts = list()
        width = cls.get_width_relation(pose)

        """
        pt1, pt2 = ClassUtils.get_rectangle_bounds_upper(pose, min_score)
        pt1 = cls._convert_pt_int(pt1)
        pt2 = cls._convert_pt_int(pt2)
        
        # First approach
        # Measure line pt
        for i in range(pt1[0] - width, pt2[0] + width):
            for j in range(pt1[1] - width, pt2[1] + width):
                if i < 0 or i >= image.shape[1] or j < 0 or j >= image.shape[0]:
                    continue

                # Check if point is contained in any rect
                contained = False

                for pt1_con, pt2_con in list_points:
                    if cls.is_contained_in_line(pt1_con, pt2_con, [i, j], width):
                        contained = True
                        break

                if contained:
                    red_val = int(image[j, i, 2])
                    green_val = int(image[j, i, 1])
                    blue_val = int(image[j, i, 0])

                    return_pts.append([red_val, green_val, blue_val])
                    if draw:
                        # Images are in bgr format
                        image[j, i] = (0, 0, 255)
        """

        # Second approach
        # Create temporal image to store points
        temp_image = np.zeros(image.shape, np.uint8)

        for pt1, pt2 in list_points:
            cls._draw_points(image, temp_image, pt1, pt2, width, return_pts, draw)

        return return_pts

    @classmethod
    def _draw_points(cls, image, temp_image, pt1, pt2, width, return_pts: list, draw):
        min_x = int(min([pt1[0], pt2[0]]))
        min_y = int(min([pt1[1], pt2[1]]))
        max_x = int(max([pt1[0], pt2[0]]))
        max_y = int(max([pt1[1], pt2[1]]))

        delta_y = pt2[1] - pt1[1]
        delta_x = pt2[0] - pt1[0]

        if abs(delta_x) > abs(delta_y):
            # Get rect eq
            # y = mx + b
            m = delta_y / delta_x
            b = pt1[1] - m * pt1[0]

            for x in range(min_x - width, max_x + 1 + width):
                if x < 0:
                    continue
                if x >= image.shape[1]:
                    continue

                y = int(m * x + b)
                for y_p in range(y - width, y + width + 1):
                    if y_p < 0:
                        continue
                    if y_p >= image.shape[0]:
                        continue
                    if temp_image[y_p, x, 0] == 255:
                        continue

                    red_val = int(image[y_p, x, 2])
                    green_val = int(image[y_p, x, 1])
                    blue_val = int(image[y_p, x, 0])

                    return_pts.append([red_val, green_val, blue_val])
                    temp_image[y_p, x, 0] = 255

                    if draw:
                        # Images are in bgr format
                        image[y_p, x] = (0, 0, 255)
        else:
            # Get rect eq
            # x = my + b
            m = delta_x / delta_y
            b = pt1[0] - m * pt1[1]

            for y in range(min_y, max_y + 1):
                if y < 0:
                    continue
                if y >= image.shape[1]:
                    continue

                x = int(m * y + b)
                for x_p in range(x - width, x + width + 1):
                    if x_p < 0:
                        continue
                    if x_p >= image.shape[1]:
                        continue
                    if temp_image[y, x_p, 0] == 255:
                        continue

                    red_val = int(image[y, x_p, 2])
                    green_val = int(image[y, x_p, 1])
                    blue_val = int(image[y, x_p, 0])

                    return_pts.append([red_val, green_val, blue_val])
                    temp_image[y, x_p, 0] = 255

                    if draw:
                        # Images are in bgr format
                        image[y, x_p] = (0, 0, 255)

    @classmethod
    def add_list_pt(cls, list_pt: list, pt1, pt2, min_score):
        if ClassUtils.check_point_integrity(pt1, min_score) and ClassUtils.check_point_integrity(pt2, min_score):
            list_pt.append((pt1, pt2))

    @classmethod
    def get_points_by_line(cls, image: np.ndarray, pt1, pt2, width, draw=False):
        # Iterate over region
        # Get histogram elems
        list_points = list()

        pt1 = cls._convert_pt_int(pt1)
        pt2 = cls._convert_pt_int(pt2)

        # Get min values
        min_x = min([pt1[0], pt2[0]])
        min_y = min([pt1[1], pt2[1]])
        max_x = max([pt1[0], pt2[0]])
        max_y = max([pt1[1], pt2[1]])

        return_pts = list()

        for i in range(min_x - width, max_x + width):
            for j in range(min_y - width, max_y + width):
                if i < 0 or i >= image.shape[1] or j < 0 or j >= image.shape[0]:
                    continue

                if cls.is_contained_in_line(pt1, pt2, [i, j], width):
                    red_val = int(image[j, i, 2])
                    green_val = int(image[j, i, 1])
                    blue_val = int(image[j, i, 0])

                    return_pts.append([red_val, green_val, blue_val])

                    if draw:
                        # Images are in bgr format
                        image[j, i] = (0, 0, 255)

        return return_pts

    @classmethod
    def is_contained_in_line(cls, pt1, pt2, pt_eval, width):
        min_x = min([pt1[0], pt2[0]])
        min_y = min([pt1[1], pt2[1]])
        max_x = max([pt1[0], pt2[0]])
        max_y = max([pt1[1], pt2[1]])

        delta_y = pt2[1] - pt1[1]
        delta_x = pt2[0] - pt1[0]

        if delta_y == 0:
            point_int = [pt_eval[0], pt1[1]]
        elif delta_x == 0:
            point_int = [pt1[0], pt_eval[1]]
        else:
            # Get slope
            m = delta_y / delta_x
            b = pt1[1] - m * pt1[0]

            # Get eq of rect
            m_per = -1 / m
            b_per = pt_eval[1] - m_per * pt_eval[0]

            # Calculate intersection
            x_int = (b_per - b)/(m - m_per)
            y_int = m * x_int + b

            point_int = [x_int, y_int]

        # Limit points
        if point_int[0] < min_x:
            point_int[0] = min_x
        if point_int[0] > max_x:
            point_int[0] = max_x
        if point_int[1] < min_y:
            point_int[1] = min_y
        if point_int[1] > max_y:
            point_int[1] = max_y

        # Get euclidean distance of points
        distance = ClassUtils.get_euclidean_distance_pt(point_int, pt_eval)

        if distance < width:
            contained = True
        else:
            contained = False

        return contained

    @staticmethod
    def _convert_pt_int(pt):
        return [int(pt[0]), int(pt[1])]

    @staticmethod
    def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=num_labels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist

    @classmethod
    def load_images_comparision_ext(cls, instance_pose: ClassOpenPose, min_score, load_one_img=False):
        print('Load image comparision')

        # Loading filename 1
        init_dir = '/home/mauricio/Pictures'
        options = {'initialdir': init_dir}
        filename1 = askopenfilename(**options)

        if not filename1:
            filename1 = '/home/mauricio/Pictures/2_1.jpg'

        ext1 = os.path.splitext(filename1)[1]
        if ext1 != '.jpg' and ext1 != '.jpeg':
            raise Exception('Extension1 is not jpg or jpeg')

        if not load_one_img:
            # Loading filename 2
            filename2 = askopenfilename(**options)
        else:
            filename2 = None

        if not filename2:
            filename2 = '/home/mauricio/Pictures/2_2.jpg'

        ext2 = os.path.splitext(filename2)[1]
        if ext2 != '.jpg' and ext2 != '.jpeg':
            raise Exception('Extension2 is not jpg or jpeg')

        image1 = cv2.imread(filename1)
        if image1 is None:
            raise Exception('Invalid image in filename {0}'.format(filename1))

        image2 = cv2.imread(filename2)
        if image2 is None:
            raise Exception('Invalid image in filename {0}'.format(filename2))

        is_json1 = True
        is_json2 = True

        new_file_1 = filename1.replace('.jpeg', '.json')
        new_file_1 = new_file_1.replace('.jpg', '.json')

        new_file_2 = filename2.replace('.jpeg', '.json')
        new_file_2 = new_file_2.replace('.jpg', '.json')

        if not os.path.exists(new_file_1):
            print('File not found: {0}'.format(new_file_1))
            is_json1 = False

        if not os.path.exists(new_file_2):
            print('File not found: {0}'.format(new_file_2))
            is_json2 = False

        if not is_json1:
            poses1 = instance_pose.recognize_image(image1)

            if len(poses1) != 1:
                raise Exception('Invalid len for pose1: {0}'.format(len(poses1)))
            if not ClassUtils.check_vector_integrity_pos(poses1[0], min_score):
                raise Exception('Pose 1 not valid')

            pose1 = poses1[0]
            label1 = 0

            color_upper1, color_lower1 = cls.process_colors_person(pose1, min_score, image1,
                                                                   decode_img=False)
        else:
            with open(new_file_1, 'r') as f:
                obj_json1 = json.loads(f.read())

            pose1 = obj_json1['vector']

            color_upper1 = obj_json1['colorUpper']
            color_lower1 = obj_json1['colorLower']

            if 'label' in obj_json1:
                label1 = obj_json1['label']
            else:
                label1 = 0

            if not ClassUtils.check_vector_integrity_pos(pose1, min_score):
                raise Exception('Pose 1 not valid')

        if not is_json2:
            poses2 = instance_pose.recognize_image(image2)

            if len(poses2) != 1:
                raise Exception('Invalid len for pose2: {0}'.format(len(poses2)))
            if not ClassUtils.check_vector_integrity_pos(poses2[0], min_score):
                raise Exception('Pose 2 not valid')

            pose2 = poses2[0]
            label2 = 0
            color_upper2, color_lower2 = cls.process_colors_person(pose2, min_score, image2,
                                                                                decode_img=False)
        else:
            with open(new_file_2, 'r') as f:
                obj_json2 = json.loads(f.read())

            pose2 = obj_json2['vector']

            color_upper2 = obj_json2['colorUpper']
            color_lower2 = obj_json2['colorLower']

            if 'label' in obj_json2:
                label2 = obj_json2['label']
            else:
                label2 = 0

            if not ClassUtils.check_vector_integrity_pos(pose2, min_score):
                raise Exception('Pose 2 not valid')

        list_points1 = cls.get_points_by_pose(image1, pose1, min_score)
        list_points2 = cls.get_points_by_pose(image2, pose2, min_score)

        return {
            'image1': image1,
            'image2': image2,
            'pose1': pose1,
            'pose2': pose2,
            'label1': label1,
            'label2': label2,
            'colorUpper1': color_upper1,
            'colorLower1': color_lower1,
            'colorUpper2': color_upper2,
            'colorLower2': color_lower2,
            'listPoints1': list_points1,
            'listPoints2': list_points2
        }

    @classmethod
    def load_images_comparision(cls, instance_pose, min_score, load_one_img=False):
        obj_json = cls.load_images_comparision_ext(instance_pose, min_score, load_one_img)
        return obj_json['image1'], obj_json['image2'], obj_json['pose1'], obj_json['pose2']
