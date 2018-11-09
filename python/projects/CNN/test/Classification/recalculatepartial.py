import math
import json
from classutils import ClassUtils
import os
import copy
from classdescriptors import ClassDescriptors
import cv2
import numpy as np

from shutil import rmtree

min_score = 0.05

list_classes = [
    {
        # Cls 0
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Door'),
    },
    {
        # Cls1
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Down'),
    },
    {
        # Cls 2
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Loitering'),
    },
    {
        # Cls 3
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Plumbs'),
    },
    {
        # Cls 4
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Squat'),
    },
    {
        # Cls 4
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Up'),
    },
    {
        # Cls 4
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Walk'),
    }
]


def main():
    print('Initializing main function')

    res = input('Press 1 to reprocess list partial: ')

    if res == '1':
        reprocess_list_partial()
    else:
        raise Exception('Option not recognized')


def reprocess_list_partial():
    print('Reprocess list partial')

    # Cleaning partial poses - Step mjpegxr
    for classInfo in list_classes:
        folder = classInfo['folderPath']
        for root, _, _ in os.walk(folder):
            if 'partial' in root and os.path.exists(root):
                print('Removing dir: {0}'.format(root))
                rmtree(root)

            if 'action_' in root and os.path.exists(root):
                print('Removing dir: {0}'.format(root))
                rmtree(root)

    # Loading zone calib info
    with open(ClassUtils.zone_calib_path, 'r') as f:
        zone_txt = f.read()

    zone_data = json.loads(zone_txt)
    
    # Checking for data in all folders and reprocess
    for classInfo in list_classes:
        folder = classInfo['folderPath']
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)

                ext = ClassUtils.get_filename_extension(full_path)
                if ext == '.json' and '_actiondata' not in file:
                    print('Processing file: {0}'.format(full_path))
                    process_list_partial(full_path, zone_data)

    # Done


def process_list_partial(filename, zone_data):
    # Loading json data from filename
    with open(filename, 'r') as f:
        person_txt = f.read()

    person = json.loads(person_txt)
    list_poses = person['listPoses']

    # Create partial list of elems
    moving = True

    index = 0
    num_poses_future = 5
    min_distance_x = 80
    min_distance_y = 80
    list_poses_action = list()
    list_poses_partial = list()

    while index < len(list_poses):
        pose = list_poses[index]
        remaining = len(list_poses) - index - 1

        pos = pose['globalPosition']

        if moving:
            valid = True
            if remaining >= num_poses_future:
                count = 0
                count_min = 0

                for i in range(index + 1, index + num_poses_future):
                    count += 1
                    pos_future = list_poses[index + num_poses_future]['globalPosition']

                    distance_x = math.fabs(pos[0] - pos_future[0])
                    distance_y = math.fabs(pos[1] - pos_future[1])
                    if distance_x < min_distance_x and distance_y < min_distance_y:
                        count_min += 1

                # All points must be in a range
                if count == count_min:
                    valid = False

            if not valid:
                # Saving current list and create a new one
                # Get position

                final_pos = list_poses[index + num_poses_future]['globalPosition']
                in_zone = is_point_in_zone(final_pos, zone_data)

                if len(list_poses_partial) != 0:

                    # Saving
                    list_poses_action.append({
                        'moving': moving,
                        'listPoses': copy.deepcopy(list_poses_partial),
                        'finalPos': final_pos,
                        'inZone': in_zone
                    })
                    list_poses_partial.clear()

                moving = False

            list_poses_partial.append(pose)
        else:
            # Generating elements into list
            valid = True
            if remaining >= num_poses_future:
                for i in range(index + 1, index + num_poses_future):
                    pos_future = list_poses[index + num_poses_future]['globalPosition']

                    distance_x = math.fabs(pos[0] - pos_future[0])
                    distance_y = math.fabs(pos[1] - pos_future[1])

                    # Inverse
                    # If there is some point with greater distance!
                    if distance_x >= min_distance_x or distance_y >= min_distance_y:
                        valid = False
                        break

            list_poses_partial.append(pose)
            if not valid:
                final_pos = list_poses[index]['globalPosition']
                in_zone = is_point_in_zone(final_pos, zone_data)

                # New pose list
                for i in range(num_poses_future):
                    list_poses_partial.append(list_poses[index + i])

                list_poses_action.append({
                    'moving': moving,
                    'listPoses': copy.deepcopy(list_poses_partial),
                    'finalPos': final_pos,
                    'inZone': in_zone
                })
                list_poses_partial.clear()

                index += num_poses_future
                moving = True

        # Iterator!
        index += 1

    if len(list_poses_partial) != 0:
        final_pos = list_poses_partial[-1]['globalPosition']
        in_zone = is_point_in_zone(final_pos, zone_data)

        # Saving elements in partial format
        list_poses_action.append({
            'moving': moving,
            'listPoses': copy.deepcopy(list_poses_partial),
            'finalPos': final_pos,
            'inZone': in_zone
        })
        list_poses_partial.clear()

    # Generating elements into list
    person_action = {
        'personGuid': person['personGuid'],
        'listPosesAction': list_poses_action
    }

    new_filename = ClassUtils.get_filename_no_extension(filename)
    ext = ClassUtils.get_filename_extension(filename)

    new_filename += '_actiondata' + ext

    action_txt = json.dumps(person_action, indent=4)
    with open(new_filename, 'w') as f:
        f.write(action_txt)

    # Saving list poses action for debugging
    save_list_poses_action(filename, list_poses_action)

    print('Done!')


def is_point_in_zone(point, zone_data):
    # Checking if point is in zone
    list_rectangles = zone_data['listRectanglePoints']

    result = False
    for pts_rect in list_rectangles:
        pt1 = pts_rect[0]
        pt2 = pts_rect[1]

        if pt1[0] <= point[0] <= pt2[0] and  pt1[0] <= point[1] <= pt2[1]:
            result = True
            break

    return result


def save_list_poses_action(filename, list_poses_action):
    folder = os.path.dirname(filename)

    for index, list_poses in enumerate(list_poses_action):
        path_actions = os.path.join(folder, 'action_' + str(index))

        if not os.path.exists(path_actions):
            os.makedirs(path_actions)

        moving = list_poses['moving']
        in_zone = list_poses['inZone']

        if moving:
            if in_zone:
                color_back = (200, 255, 255)
            else:
                color_back = (255, 255, 255)
        else:
            if in_zone:
                color_back = (255, 200, 255)
            else:
                color_back = (255, 255, 200)

        for pose in list_poses['listPoses']:
            transformed_points = pose['transformedPoints']
            global_position = pose['globalPosition']
            key_pose = pose['keyPose']
            ticks = pose['ticks']

            re_scale_factor = 100
            scaled_points = ClassDescriptors.re_scale_pose_factor(transformed_points, re_scale_factor, min_score)

            person_array = list()
            person_array.append([0, 0, 0])
            person_array.append([0, 0, 1])
            for point in scaled_points:
                person_array.append([point[0], point[1], 1])
            for _ in range(10):
                person_array.append([0, 0, 0])

            local_position = ClassDescriptors.get_local_position_point(person_array, min_score, key_pose)

            # Draw global position
            min_x = -200
            max_x = 1000

            min_y = -900
            max_y = 1200

            delta_x = max_x - min_x
            delta_y = max_y - min_y

            width_plane = 1000
            height_plane = 500

            width_rect = 10

            img_plane = np.zeros((height_plane, width_plane, 3), np.uint8)

            new_person_array = list()

            # Height plane must be set into list
            pt_plane_x = int((global_position[0] - min_x) * width_plane / delta_x)
            pt_plane_y = height_plane - int((global_position[1] - min_y) * height_plane / delta_y)

            pt1 = (pt_plane_x - width_rect, pt_plane_y - width_rect)
            pt2 = (pt_plane_x + width_rect, pt_plane_y + width_rect)

            delta_x_pos = pt_plane_x - local_position[0]
            delta_y_pos = pt_plane_y - local_position[1]

            # Blank image
            img_plane[:, :] = color_back

            # Transform local vector coordinates
            for point in person_array:
                if ClassUtils.check_point_integrity(point, min_score):
                    new_person_array.append([point[0] + delta_x_pos, point[1] + delta_y_pos, 1])
                else:
                    new_person_array.append(point)

            # Draw pose
            ClassDescriptors.draw_pose(img_plane, new_person_array, min_score, key_pose)

            # Fill rectangle
            cv2.rectangle(img_plane, pt1, pt2, (255, 0, 255), thickness=-1)

            # Save global position info
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (0, 0, 0)
            line_type = 2

            cv2.putText(img_plane, '({0:.2f}, {1:.2f})'.format(global_position[0], global_position[1]), pt2,
                        font, font_scale, font_color, line_type)

            filename = os.path.join(path_actions, '{0}_g.jpg'.format(ticks))
            cv2.imwrite(filename, img_plane)


if __name__ == '__main__':
    main()
