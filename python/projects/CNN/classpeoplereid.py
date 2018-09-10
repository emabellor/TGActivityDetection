from datetime import datetime
from classutils import ClassUtils
import uuid
import math


class ClassPeopleReId:

    def __init__(self, _person_param, _date_ref, _person_guid=''):
        self.person_param = _person_param

        self.last_date = _date_ref

        if _person_guid == '':
            self.person_guid = str(uuid.uuid4())
        else:
            self.person_guid = _person_guid

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
    def cam_number(self):
        return self.person_param['camNumber']

    @property
    def pose_guid(self):
        return self.person_param['poseGuid']

    def get_guid_relation(self):
        return {
            'personGuid': self.person_guid,
            'poseGuid': self.pose_guid
        }

    @classmethod
    def load_people_from_frame_info(cls, frame_info, date_ref):
        list_people = list()

        params = frame_info[2]['params']

        for param in params:
            integrity = param['integrity']

            if integrity:
                list_people.append(cls(param, date_ref))

        return list_people

    @classmethod
    def load_list_people(cls, frame_info_list, date_ref):
        list_people = list()

        for frame_info in frame_info_list:
            list_people = frame_info[2]['listPeople']
            params = frame_info[2]['params']

            for person in list_people:
                pose_guid = person['poseGuid']
                person_guid = person['personGuid']

                found = False
                for param in params:
                    if param['pose_guid'] == pose_guid:
                        found = True
                        list_people.append(cls(param, date_ref, person_guid))
                        break

                if not found:
                    raise Exception('Cant find pose guid {0}'.format(pose_guid))

        return list_people

    def update_values_from_person(self, person, date_ref):
        self.person_param = person.person_param
        self.last_date = date_ref

    @classmethod
    def get_people_diff(cls, person1: 'ClassPeopleReId', person2: 'ClassPeopleReId'):
        # Comparing people between list
        diff_upper = ClassUtils.get_color_diff_rgb(person1.color_upper, person2.color_upper)
        diff_lower = ClassUtils.get_color_diff_rgb(person1.color_lower, person2.color_lower)

        diff_colors_1 = ClassUtils.get_color_diff_rgb(person1.color_upper, person1.color_lower)
        diff_colors_2 = ClassUtils.get_color_diff_rgb(person2.color_upper, person2.color_lower)

        diff_colors = math.fabs(diff_colors_1 - diff_colors_2)

        distance = cls.get_person_distance(person1, person2)

        return_data = {
            'diffUpper': diff_upper,
            'diffLower': diff_lower,
            'diffColors': diff_colors,
            'distance': distance
        }

        return return_data

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

        # Normalize distance between 0 and 500 - If greater, diff is one
        norm_distance = distance
        if norm_distance > 500:
            norm_distance = 500

        norm_distance /= 500

        return norm_upper, norm_lower, norm_distance

    @classmethod
    def compare_people(cls, person1: 'ClassPeopleReId', person2: 'ClassPeopleReId'):
        norm_upper, norm_lower, norm_distance = cls.compare_people_items(person1, person2)

        # Get mean from distance
        color_person = (norm_upper + norm_lower) / 2

        # Generating color descriptor from list
        # Inspired in Jang et al
        # Score is descending
        alpha = 0.7
        score = alpha * color_person + (1 - alpha) * norm_distance

        return score

    @staticmethod
    def get_person_distance(person1: 'ClassPeopleReId', person2: 'ClassPeopleReId'):
        # Comparing person distance
        distance = ClassUtils.get_euclidean_distance(person1.global_pos[0], person1.global_pos[1],
                                                     person2.global_pos[0], person2.global_pos[1])
        return distance

    def get_rgb_color(self):
        # Getting RGB color from GUID
        uuid_bin = uuid.UUID(self.person_guid).bytes

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

