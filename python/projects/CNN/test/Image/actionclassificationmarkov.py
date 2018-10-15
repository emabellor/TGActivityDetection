import os
from classutils import ClassUtils
from classhmm import ClassHMM
import random
import cv2
import json
import numpy as np

seed = 1234
hmm_models = list()
hidden_states = 10
iterations = 20

list_folder_data = [
    {
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Door'),
        'label': 0
    },
    {
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Moving'),
        'label': 1
    },
    {
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Squat'),
        'label': 2
    },
    {
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Quiet'),
        'label': 3
    },
    {
        'folderPath': os.path.join(ClassUtils.cnn_base_path, 'Classes/No_Mov/Plumbs'),
        'label': 4
    }
]


def main():
    print('Initializing main function')
    classify_markov()


def classify_markov():
    global hmm_models

    print('Init classification using markov')
    training_data = list()
    training_labels = list()
    training_files = list()

    eval_data = list()
    eval_labels = list()
    eval_files = list()

    # Loading classes
    for index, item in enumerate(list_folder_data):
        folder = item['folderPath']
        label = item['label']

        num_file = 0
        list_paths = list()
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)

                ext = ClassUtils.get_filename_extension(full_path)

                if ext == '.json':
                    list_paths.append(full_path)

        total_samples = len(list_paths)
        total_train = int(total_samples * 80 / 100)

        # Shuffle samples
        random.Random(seed).shuffle(list_paths)

        for full_path in list_paths:
            list_key_poses = list()

            with open(full_path, 'r') as f:
                json_txt = f.read()

            json_data = json.loads(json_txt)
            list_poses = json_data['listPoses']

            for pose in list_poses:
                list_key_poses.append(pose['keyPose'])

            if num_file < total_train:
                training_data.append(list_key_poses)
                training_labels.append(label)
                training_files.append(full_path)
            else:
                eval_data.append(list_key_poses)
                eval_labels.append(label)
                eval_files.append(full_path)

            num_file += 1

    print('Total training: {0}'.format(len(training_data)))
    print('Total eval: {0}'.format(len(eval_data)))

    res = input('Press 1 to train. Press 2 to eval - Press 3 to train iter: ')

    if res == '1':
        train_markov(training_data, training_labels, training_files, eval_data, eval_labels)
    elif res == '2':
        # Pre-loading models
        for i in range(len(list_folder_data)):
            model_path = os.path.join(ClassHMM.model_hmm_folder, 'model{0}.pkl'.format(i))
            hmm_model = ClassHMM(model_path)
            hmm_models.append(hmm_model)

        eval_markov(eval_data, eval_labels)
    elif res == '3':
        train_markov_iter(training_data, training_labels, training_files, eval_data, eval_labels)
    else:
        raise Exception('Option not implemented: {0}'.format(res))


def train_markov(training_data, training_labels, training_files, eval_data, eval_labels):
    global hmm_models
    global hidden_states
    print('Init training markov')

    cls = 0
    while True:
        list_data = list()
        list_files = list()

        for index in range(len(training_data)):
            if training_labels[index] == cls:
                seq_list = training_data[index]
                file = training_files[index]

                list_data.append(seq_list)
                list_files.append(file)

        if len(list_data) == 0:
            break
        else:
            model_path = os.path.join(ClassHMM.model_hmm_folder, 'model{0}.pkl'.format(cls))
            hmm_model = ClassHMM(model_path)

            print('Training model {0}'.format(cls))
            hmm_model.train(list_data, hidden_states)
            hmm_models.append(hmm_model)
            cls += 1

    print('Models trained')
    eval_markov(eval_data, eval_labels)


def train_markov_iter(training_data, training_labels, training_files, eval_data, eval_labels):
    global hmm_models
    global hidden_states
    print('Init training markov iteration')

    selected_models = list()

    max_precision = 0
    for _ in range(iterations):
        # Reset iterations
        hmm_models.clear()
        cls = 0

        while True:
            list_data = list()
            list_files = list()

            for index in range(len(training_data)):
                if training_labels[index] == cls:
                    seq_list = training_data[index]
                    file = training_files[index]

                    list_data.append(seq_list)
                    list_files.append(file)

            if len(list_data) == 0:
                break
            else:
                model_path = os.path.join(ClassHMM.model_hmm_folder, 'model{0}.pkl'.format(cls))
                hmm_model = ClassHMM(model_path)

                print('Training model {0}'.format(cls))
                hmm_model.train(list_data, hidden_states)
                hmm_models.append(hmm_model)
                cls += 1

        print('Models trained')
        precision = eval_markov(eval_data, eval_labels)

        if precision > max_precision:
            selected_models.clear()

            # Copy list no reference
            for model in hmm_models:
                selected_models.append(model)

            max_precision = precision

    # Saving selected models
    hmm_models = selected_models
    for model in selected_models:
        model.save_model()

    print('Evaluating model for final precision: {0}'.format(max_precision))
    eval_markov(eval_data, eval_labels)

    print('Evaluating model second chance')
    eval_markov(eval_data, eval_labels)
    print('Done!')


def eval_markov(eval_data, eval_labels):
    print('Init eval markov')

    count = 0

    # Getting confussion matrix
    print('Getting confusion matrix')
    classes = len(list_folder_data)

    confusion_np = np.zeros((classes, classes))
    for index in range(len(eval_labels)):
        data = eval_data[index]
        label = eval_labels[index]

        predict = predict_data(data)

        if label == predict:
            count += 1

        confusion_np[label, predict] += 1

    precision = count / len(eval_labels)

    print('Precision: {0}'.format(precision))
    print('Confussion Matrix')
    print(confusion_np)
    return precision


def predict_data(list_poses):
    global hmm_models

    max_score = 0
    cls = 0
    first = True
    for index, model in enumerate(hmm_models):
        score = model.get_score(list_poses)
        if first:
            max_score = score
            cls = index
            first = False
        else:
            if score > max_score:
                max_score = score
                cls = index

    return cls


if __name__ == '__main__':
    main()

