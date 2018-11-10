from classutils import ClassUtils
import json
import os
import random
from classhmm import ClassHMM
import numpy as np

seed = 1234

list_classes = [
    {
        # Cls 0
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Door'),
    },
    {
        # Cls 1
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
        # Cls 5
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Up'),
    },
    {
        # Cls 6
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Walk'),
    }
]


def main():
    print('Initializing main function')

    # Loading elements into list
    training_list_actions = list()
    training_labels = list()
    eval_list_actions = list()
    eval_labels = list()

    for index, item in enumerate(list_classes):
        folder = item['folderPath']
        label = index

        list_list_actions = list()
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                extension = ClassUtils.get_filename_extension(full_path)

                if extension == '.json' and '_actiondata' in full_path:
                    with open(full_path, 'r') as f:
                        json_txt = f.read()

                    json_data = json.loads(json_txt)
                    list_list_actions.append(json_data['listActions'])

        total_samples = len(list_list_actions)
        total_train = int(total_samples * 80 / 100)
        print('Total samples: {0}'.format(total_samples))

        # Shuffle samples
        random.Random(seed).shuffle(list_list_actions)

        num_file = 0
        for list_poses in list_list_actions:
            if num_file < total_train:
                training_list_actions.append(list_poses)
                training_labels.append(label)
            else:
                eval_list_actions.append(list_poses)
                eval_labels.append(label)

            num_file += 1

    res = input('Press 1 to train HMM: ')

    if res == '1':
        train_hmm(training_list_actions, training_labels, eval_list_actions, eval_labels)
    else:
        raise Exception('Raise Exception!')


def train_hmm(training_list_actions, training_labels, eval_list_actions, eval_labels):
    print('Initializing training HMM')
    hmm_models = list()
    hidden_states = 4

    for index in range(len(list_classes)):
        # Creating model for each class
        model_path = os.path.join(ClassHMM.model_hmm_folder_activities, 'model{0}.pkl'.format(index))
        hmm_model = ClassHMM(model_path)

        # Get sequences for class:
        list_data = list()
        for idx_label, label in enumerate(training_labels):
            if label == index:
                list_actions = training_list_actions[idx_label]

                seq = list()
                for action in list_actions:
                    seq.append(action['class'])

                list_data.append(seq)

        print('Training model {0}'.format(index))
        hmm_model.train(list_data, hidden_states)
        hmm_models.append(hmm_model)

    print('Evaluating Markov!')
    eval_markov(eval_list_actions, eval_labels, hmm_models)


def eval_markov(eval_list_actions, eval_labels, hmm_models):
    print('Init eval markov')

    count = 0

    # Getting confussion matrix
    print('Getting confusion matrix')
    classes = len(hmm_models)

    confusion_np = np.zeros((classes, classes))
    for index in range(len(eval_labels)):
        list_actions = eval_list_actions[index]

        seq = list()
        for action in list_actions:
            seq.append(action['class'])

        label = eval_labels[index]

        predict = predict_data(seq, hmm_models)

        if label == predict:
            count += 1

        confusion_np[label, predict] += 1

    precision = count / len(eval_labels)

    print('Precision: {0}'.format(precision))
    print('Confussion Matrix')
    print(confusion_np)
    return precision


def predict_data(list_poses, hmm_models):
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
