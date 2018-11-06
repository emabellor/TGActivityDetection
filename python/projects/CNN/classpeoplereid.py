from datetime import datetime
from classutils import ClassUtils
from classdescriptors import ClassDescriptors
import uuid
import math
import copy


class ClassPeopleReId:

    def __init__(self, _person_param, _date_ref, _person_guid=''):
        self.person_param = _person_param

        self.last_date = _date_ref
        self.list_poses = list()

        if _person_guid == '':
            self.person_guid = str(uuid.uuid4())
        else:
            self.person_guid = _person_guid

        ticks = ClassUtils.datetime_to_ticks(_date_ref)
        self._add_pose_to_list(ticks, _person_param)

        self.update_counter = 0

    @property
    def vectors(self):
        return self.person_param['vectors']

    @property
    def local_pos(self):
        return self.person_param['localPosition']

    @property
    def global_pos(self):
        return self.person_param['globalPosition']

    @property
    def color_upper(self):
        return self.person_param['colorUpper']

    @property
    def color_lower(self):
        return self.person_param['colorLower']

    @property
    def hist_pose(self):
        return self.person_param['histPose']

    @property
    def cam_number(self):
        return self.person_param['camNumber']

    @property
    def pose_guid(self):
        return self.person_param['poseGuid']

    @property
    def only_pos(self):
        return self.person_param['onlyPos']

    @property
    def transformed_points(self):
        return self.person_param['transformedPoints']

    def get_guid_relation(self):
        return {
            'personGuid': self.person_guid,
            'poseGuid': self.pose_guid
        }

    def _add_pose_to_list(self, ticks, param):
        self.list_poses.append({
            'transformedPoints': param['transformedPoints'],
            'ticks': ticks,
            'keyPose': param['keyPose'],
            'probability': param['probability'],
            'poseGuid': param['poseGuid'],
            'globalPosition': param['globalPosition'],
            'localPosition': param['localPosition'],
            'vectors': param['vectors'],
            'angles': param['angles']
        })

    @classmethod
    def load_people_from_frame_info(cls, frame_info, date_ref):
        list_people = list()

        params = frame_info[2]['params']

        # Deep copy params to avoid problems
        params_cpy = copy.deepcopy(params)

        found = frame_info[2]['found']

        ticks = ClassUtils.datetime_to_ticks(date_ref)
        counter = 0
        for param in params_cpy:
            cam_number = param['camNumber']
            integrity = param['integrity']

            # Add elements using valid skeletons
            # Ignore skeletons marked with only pos element
            if integrity and found:
                person_guid = '{0}_{1}_{2}'.format(cam_number, counter, ticks)
                list_people.append(cls(param, date_ref, _person_guid=person_guid))
                counter += 1

        return list_people

    @classmethod
    def load_list_people(cls, frame_info_list, date_ref):
        # Only for debugging purposes!
        list_people = list()

        for frame_info in frame_info_list:
            params = frame_info[2]['params']

            for param in params:
                person_guid = param['personGuid']

                if person_guid != '':
                    list_people.append(cls(param, date_ref, person_guid))

        return list_people

    def update_values_from_person(self, person, date_ref):
        self.person_param = person.person_param
        self.last_date = date_ref

        ticks = ClassUtils.datetime_to_ticks(date_ref)
        self._add_pose_to_list(ticks, person.person_param)
        self.update_counter = 0

    def set_not_updated(self):
        self.update_counter += 1

    @classmethod
    def get_people_diff(cls, person1: 'ClassPeopleReId', person2: 'ClassPeopleReId'):
        # Comparing people between list
        diff_upper = ClassUtils.get_color_diff_rgb(person1.color_upper, person2.color_upper)
        diff_lower = ClassUtils.get_color_diff_rgb(person1.color_lower, person2.color_lower)

        diff_colors_1 = ClassUtils.get_color_diff_rgb(person1.color_upper, person1.color_lower)
        diff_colors_2 = ClassUtils.get_color_diff_rgb(person2.color_upper, person2.color_lower)

        diff_colors = math.fabs(diff_colors_1 - diff_colors_2)

        diff_k_means = ClassDescriptors.get_kmeans_diff(person1.hist_pose, person2.hist_pose)

        distance = cls.get_person_distance(person1, person2)

        return_data = {
            'diffUpper': diff_upper,
            'diffLower': diff_lower,
            'diffColors': diff_colors,
            'distance': distance,
            'diffKMeans': diff_k_means
        }

        return return_data

    @classmethod
    def compare_people_kmeans(cls, person1: 'ClassPeopleReId', person2: 'ClassPeopleReId'):
        return_data = cls.get_people_diff(person1, person2)

        diff_kmeans = return_data['diffKMeans']
        distance = return_data['distance']

        # Change distance for velocity
        print('Comparing elements with stamps {0} - {1}'.format(person1.last_date, person2.last_date))
        delta_time = math.fabs((person1.last_date - person2.last_date).total_seconds())
        if delta_time == 0:
            print('Hello!')

        norm_k = diff_kmeans
        if norm_k > 50:
            norm_k = 50
        norm_k /= 50

        velocity = distance / delta_time

        # Delta time for noise
        if delta_time < 0.51:
            if velocity > 350:
                velocity = 350
            velocity /= 350
        else:
            if velocity > 250:
                velocity = 250
            velocity /= 250

        return norm_k, velocity

    @classmethod
    def compare_people_items(cls, person1: 'ClassPeopleReId', person2: 'ClassPeopleReId'):
        # Comparing people between list
        diff_upper = ClassUtils.get_color_diff_rgb(person1.color_upper, person2.color_upper)
        diff_lower = ClassUtils.get_color_diff_rgb(person1.color_lower, person2.color_lower)
        distance = cls.get_person_distance(person1, person2)

        # Color diff between 1 and 100
        # Distance in cm

        # Normalize diffs between 0 and 50 - If greater, diff is one
        norm_upper = diff_upper
        if norm_upper > 50:
            norm_upper = 50
        norm_upper /= 50

        norm_lower = diff_lower
        if norm_lower > 50:
            norm_lower = 50
        norm_lower /= 50

        # Change distance for velocity
        delta_time = math.fabs((person1.last_date - person2.last_date).total_seconds())
        if delta_time == 0:
            print('Hello!')

        velocity = distance / delta_time
        if velocity > 200:
            velocity = 200

        velocity /= 200

        return norm_upper, norm_lower, velocity

    @classmethod
    def compare_people(cls, person1: 'ClassPeopleReId', person2: 'ClassPeopleReId'):
        norm_upper, norm_lower, norm_velocity = cls.compare_people_items(person1, person2)

        # Get mean from distance
        color_person = (norm_upper + norm_lower) / 2

        # Generating color descriptor from list
        # Inspired in Jang et al
        # Score is descending
        alpha = 0.7
        score = alpha * color_person + (1 - alpha) * norm_velocity

        return score

    @staticmethod
    def get_person_distance(person1: 'ClassPeopleReId', person2: 'ClassPeopleReId'):
        # Comparing person distance
        distance = ClassUtils.get_euclidean_distance(person1.global_pos[0], person1.global_pos[1],
                                                     person2.global_pos[0], person2.global_pos[1])
        return distance

    def get_rgb_color(self):
        # Getting RGB color from GUID
        uuid_bin = self.person_guid.encode('utf-8')

        # Return first bytes to get color
        return uuid_bin[0], uuid_bin[1], uuid_bin[2]

    def get_bgr_color(self):
        r, g, b = self.get_rgb_color()
        return b, g, r

    def get_rgb_color_str(self):
        r, g, b = self.get_rgb_color()
        arr_color = [r, g, b]

        return bytes(arr_color).hex()

    def get_rgb_color_str_int(self):
        r, g, b = self.get_rgb_color()
        return '{0},{1},{2}'.format(r, g, b)

