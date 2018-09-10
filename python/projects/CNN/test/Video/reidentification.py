from datetime import datetime
from datetime import timedelta
from classmjpegdate import ClassMjpegDate
from classutils import ClassUtils
import uuid
import logging
import os
import numpy as np
import cv2
import shutil

min_distance = 1000
clean_timeout_sec = 5
game_period_ms = 500

logger = logging.getLogger('main')
CNN_BASE_FOLDER = '/home/mauricio/CNN/Images'


def main():
    print('Warning: This code is deprecated')
    print('Use testconvertmjpegxr.py instead')
    FORMAT = "%(asctime)s [%(name)-16.16s] [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    print('Init re-id')

    # Cleaning image tree
    clean_base_folder()

    list_cams = [419, 420, 421, 428, 429, 430]
    date_init = datetime(2018, 2, 24, 14, 15, 0)
    date_end = datetime(2018, 2, 24, 15, 15, 0)

    date_video = date_init
    list_readers = []
    for i in range(len(list_cams)):
        list_readers.append(ClassMjpegDate(list_cams[i]))

    list_people = []

    while date_video < date_end:
        print(date_video)
        for i in range(len(list_cams)):
            frame_info = list_readers[i].load_frame(date_video)

            dic_frame = frame_info[2]
            image = frame_info[0]

            frame_position = dic_frame['positions']
            vectors = dic_frame['vectors']
            guid_list = []

            for candidate_position in frame_position:
                guid = ''
                score = candidate_position[2]
                index = frame_position.index(candidate_position)
                person_vectors = vectors[index]

                # Only takes into account scores in one
                if score == 1:

                    found = False
                    person_candidate = None

                    for person in list_people:
                        if not person['updated']:
                            person_position = person['position']
                            distance = ClassUtils.get_euclidean_distance(candidate_position[0], candidate_position[1],
                                                                         person_position[0], person_position[1])

                            if distance < min_distance:
                                found = True
                                person_candidate = person
                                break

                    if not found:
                        # Creates candidate into list
                        candidate = create_candidate(candidate_position, date_video, person_vectors, image)
                        list_people.append(candidate)
                        guid = candidate['guid']
                        print('Creating guid {0}'.format(guid))
                    else:
                        # Update only position
                        update_candidate(person_candidate, candidate_position, date_video, person_vectors, image)
                        guid = person_candidate['guid']
                        print('Updating guid {0}'.format(guid))

                guid_list.append(guid)

            # Clean old records -> 5 seconds
            people_to_remove = []
            for person in list_people:
                last_update = person['last_update']

                delta_update = date_video - last_update

                if delta_update.total_seconds() > clean_timeout_sec:
                    people_to_remove.append(person)

            for person in people_to_remove:
                # Remove from list and process action
                list_people.remove(person)
                get_cnn_images(person)

            # Update person info again
            for person in list_people:
                person['updated'] = False

            # Adding element to dict_frame
            dic_frame['guids'] = guid_list
            list_readers[i].update_frame(frame_info)

        # Adding time to list
        date_video = date_video + timedelta(milliseconds=game_period_ms)

    # Done in all cycles -> Saving results
    for reader in list_readers:
        reader.save_frame_info()

    # Done


def clean_base_folder():
    shutil.rmtree(CNN_BASE_FOLDER)


def get_cnn_images(person):
    # Getting cnn images
    # Frame list must be ordered in list frames
    # Custom frames between person
    # Save vector, position
    list_frames = person['list_frames']
    new_person_vectors = []

    for frame in list_frames:
        person_vectors = frame['person_vectors']

        torso_dis_1 = ClassUtils.get_euclidean_distance(person_vectors[1][0], person_vectors[1][1],
                                                        person_vectors[8][0], person_vectors[8][1])

        torso_dis_2 = ClassUtils.get_euclidean_distance(person_vectors[1][0], person_vectors[1][1],
                                                        person_vectors[11][0], person_vectors[11][1])

        logger.debug('Distance 1: {0} Distance 2: {1}'.format(torso_dis_1, torso_dis_2))
        torso_dis = (torso_dis_1 + torso_dis_2) / 2

        print('Generating normalization')

        neck_vector = person_vectors[1]

        # Ignoring vectors from face
        # Only consider vectors from 1 to 14
        new_vectors = []
        for i in range(14):
            vector = person_vectors[i]
            score = vector[2]

            if score > ClassUtils.MIN_POSE_SCORE:
                new_x = (vector[0] - neck_vector[0]) / torso_dis
                new_y = (vector[1] - neck_vector[1]) / torso_dis
                new_vectors.append([new_x, new_y])
            else:
                new_x = 0
                new_y = 0
                new_vectors.append([new_x, new_y])

        new_person_vectors.append(new_vectors)

    # Get min_x, max_x, min_y, max_y of the sequence
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0

    for new_vectors in new_person_vectors:
        for vector in new_vectors:
            if vector[0] < min_x:
                min_x = vector[0]
            if vector[0] > max_x:
                max_x = vector[0]

            if vector[1] < min_y:
                min_y = vector[1]
            if vector[1] > max_y:
                max_y = vector[1]

    # Normalizing vectors between min and max
    factor_x = 255 / (max_x - min_x)
    factor_y = 255 / (max_y - min_y)

    for new_vectors in new_person_vectors:
        for vector in new_vectors:
            vector[0] = int((vector[0] - min_x) * factor_x)
            vector[1] = int((vector[1] - min_y) * factor_y)

    # Creating image based on opencv library
    rows = len(new_person_vectors[0])
    cols = len(new_person_vectors)

    blank_image = np.zeros((rows, cols, 3), np.uint8)

    for j in range(cols):
        for i in range(rows):
            # First item are frame, second item are joint position
            # Image rows are joint position
            # Image cols are frame
            x_value = new_person_vectors[j][i][0]
            y_value = new_person_vectors[j][i][1]
            blank_image[i, j] = [x_value, y_value, 0]

    # Creating folder from videos
    if not os.path.isdir(CNN_BASE_FOLDER):
        os.makedirs(CNN_BASE_FOLDER)

    # Saving image as jpeg
    path_image = os.path.join(CNN_BASE_FOLDER, person['guid'] + '.jpg')
    cv2.imwrite(path_image, blank_image)

    # Saving image list for debug purposes
    for frame in list_frames:
        index_frame = list_frames.index(frame)
        image = frame['image']

        image_folder = os.path.join(CNN_BASE_FOLDER, person['guid'])
        if not os.path.isdir(image_folder):
            os.makedirs(image_folder)

        aux_image_path = os.path.join(image_folder, str(index_frame) + '.jpg')
        ClassUtils.write_bin_to_file(aux_image_path, image)

    print('Done!')


def create_candidate(position, date_video, person_vectors, image):
    # Creating candidate using position and guid
    last_update = date_video
    guid = str(uuid.uuid4())

    list_frames = [{
        'person_vectors': person_vectors,
        'image': image
    }]

    candidate = {
        'position': position,
        'guid': guid,
        'last_update': last_update,
        'updated': True,  # Necessary to make update control
        'list_frames': list_frames
    }

    return candidate


def update_candidate(candidate, position, date_video, person_vectors, image):
    # Updating candidate
    candidate['position'] = position
    candidate['last_update'] = date_video
    candidate['updated'] = True
    candidate['list_frames'].append({
        'person_vectors': person_vectors,
        'image': image
    })


if __name__ == '__main__':
    main()
