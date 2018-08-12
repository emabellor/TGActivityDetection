from classutils import ClassUtils
from enum import Enum
import math


# Static class
class ClassDescriptors:

    @staticmethod
    def get_person_descriptors(person_array, min_pose_score):
        # Person descriptors for vector integrity
        # In function ClassUtils.check_vector_integrity_part
        # One part of the arms and one of the legs must exist
        # Vectors 11 and 14 are optional
        # Vectors of the torso must exist

        relation = ClassDescriptors._get_torso_shoulders_relation(person_array, min_pose_score)
        list_angles, list_angles_degrees = ClassDescriptors._get_person_descriptors_angles(person_array, min_pose_score)
        transformed_points = ClassDescriptors._get_transformed_points(person_array, min_pose_score)

        full_desc = list_angles
        full_desc += ClassUtils.get_flat_list(transformed_points)

        result = {
            'relation': relation,
            'angles': list_angles,
            'angles_degrees': list_angles_degrees,
            'transformed_points': transformed_points,
            'full_desc': full_desc
        }

        return result

    @staticmethod
    def _get_person_descriptors_angles(person_array, min_pose_score):
        # Function compatible with check vector integrity
        # In total we have 8 angles
        # and 4 angles related with torso and shoulders

        # Depends of the integrity of the points listed in list
        points_arm_right = [person_array[2], person_array[3], person_array[4]]
        points_arm_left = [person_array[5], person_array[6], person_array[7]]

        points_leg_right = [person_array[9], person_array[10], person_array[11]]
        points_leg_left = [person_array[12], person_array[13], person_array[14]]
        points_leg_partial = [person_array[9], person_array[10], person_array[12], person_array[13]]

        # Arms angles
        # If both arms are valid iterate from each point
        # If one of the arms is valid iterate and repeat
        list_angles = list()
        list_angles_degrees = list()

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

        if ClassUtils.check_point_list(points_arm_right, min_pose_score) \
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
    def _get_torso_shoulders_relation(person_array, min_pose_score):
        # Function compatible with check_vector_integrity_part
        # Depends of vector integrity of shoulders and arms
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
    def _get_transformed_points(person_array, min_pose_score):
        list_points = list()

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

        if ClassUtils.check_point_list(points_arm_right, min_pose_score) \
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

