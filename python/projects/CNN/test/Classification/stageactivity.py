from classutils import ClassUtils
import json
import os
import random
from classhmm import ClassHMM
import numpy as np
from classnn import ClassNN
from enum import Enum


class Option(Enum):
    HMM = 1
    BOW = 2


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
    res = input('Press 1 to train HMM - 2 to train bow: ')

    if res == '1':
        load_and_train(Option.HMM)
    elif res == '2':
        load_and_train(Option.BOW)
    else:
        raise Exception('Option not recognized: {0}'.format(res))


def load_and_train(option: Option):
    base_data_1 = 1
    base_data_2 = 1

    found_1 = False
    while True:
        # Loading elements into list
        training_list_actions = list()
        training_labels = list()

        eval_list_actions = list()
        eval_labels = list()

        training_list_cls = list()
        training_cls_labels = list()

        validate_list_cls = list()
        validate_cls_labels = list()

        found_2 = False

        for index, item in enumerate(list_classes):
            folder = item['folderPath']
            label = index

            list_list_actions = list()
            list_files = list()
            for root, _, files in os.walk(folder):
                for file in files:
                    full_path = os.path.join(root, file)

                    if '{0}_{1}_actiondata'.format(base_data_1, base_data_2) in full_path:
                        list_files.append(full_path)
                        found_1 = True
                        found_2 = True

            # Avoid mis_order
            list_files.sort()
            # Use static seed to ensure same order
            random.Random(seed).shuffle(list_files)

            for full_path in list_files:
                with open(full_path, 'r') as f:
                    json_txt = f.read()

                json_data = json.loads(json_txt)
                list_list_actions.append(json_data['listActions'])

            total_samples = len(list_list_actions)
            total_train = int(total_samples * 80 / 100)
            total_train_cls = int(total_samples * 60 / 100)
            print('Total samples: {0}'.format(total_samples))

            num_file = 0
            for list_actions in list_list_actions:
                if num_file < total_train:
                    training_list_actions.append(list_actions)
                    training_labels.append(label)

                    # Add cls list for markov models
                    if num_file < total_train_cls:
                        training_list_cls.append(list_actions)
                        training_cls_labels.append(label)
                    else:
                        validate_list_cls.append(list_actions)
                        validate_cls_labels.append(label)
                else:
                    eval_list_actions.append(list_actions)
                    eval_labels.append(label)

                num_file += 1

        if not found_2:
            if found_1:
                base_data_1 += 1
                base_data_2 = 1
                found_1 = False
                continue
            else:
                # There is no data for base_cls - Breaking!
                break

        print('Processing data for {0}_{1}'.format(base_data_1, base_data_2))
        if option == Option.HMM:
            train_hmm(training_list_actions, training_labels, eval_list_actions, eval_labels, option)
        elif option == Option.BOW:
            train_bow(training_list_cls, training_cls_labels,
                      validate_list_cls, validate_cls_labels,
                      eval_list_actions, eval_labels,
                      option)
        else:
            raise Exception('Raise Exception!')

        base_data_2 += 1

    print('No found more elements!')
    print('Done!')


def train_hmm(training_list_actions, training_labels, eval_list_actions, eval_labels, option: Option):
    print('Initializing training HMM')
    hmm_models = list()
    hidden_states = 7

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
                    count = action['count']
                    cls = action['class']

                    if cls == 8 or cls == 9 or cls == 10 or cls == 11:
                        # Ignore time
                        seq.append(cls)
                    else:
                        # Add time information
                        for _ in range(count):
                            seq.append(cls)

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


def train_bow(training_list_cls, training_cls_labels,
              validate_list_cls, validate_cls_labels,
              eval_list_actions, eval_labels,
              option: Option):
    print('Training BoW')

    # Generating BoW descriptors
    train_descriptors = list()
    for list_actions in training_list_cls:
        words = get_bow_descriptors(list_actions)
        train_descriptors.append(words)

    descriptors_np = np.asanyarray(train_descriptors, dtype=np.float)
    training_labels_np = np.asanyarray(training_cls_labels, dtype=np.int)

    # Generating instance_nn and train model
    cls_number = len(list_classes)
    hidden_neurons = 20
    instance_nn = ClassNN(ClassNN.model_dir_activity, cls_number, hidden_neurons)
    instance_nn.train_model(descriptors_np, training_labels_np)

    # Validating model
    validate_descriptors = list()
    for list_actions in validate_list_cls:
        words = get_bow_descriptors(list_actions)
        validate_descriptors.append(words)

    validate_descriptors_np = np.asanyarray(validate_descriptors, dtype=np.float)
    validate_labels_np = np.asanyarray(validate_cls_labels, dtype=np.int)

    accuracy = instance_nn.eval_model(validate_descriptors_np, validate_labels_np)
    print('Local accuracy: {0}'.format(accuracy))

    # Evaluating
    eval_descriptors = list()
    for list_actions in eval_list_actions:
        words = get_bow_descriptors(list_actions)
        eval_descriptors.append(words)

    eval_descriptors_np = np.asanyarray(eval_descriptors, dtype=np.float)
    eval_labels_np = np.asanyarray(eval_labels, dtype=np.int)

    accuracy = instance_nn.eval_model(eval_descriptors_np, eval_labels_np)
    print('Real accuracy: {0}'.format(accuracy))

    # Apply classification for each item!
    for item in list_classes:
        folder = item['folderPath']
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)

                if '_actiondata' in full_path:
                    with open(full_path, 'r') as f:
                        json_txt = f.read()

                    json_data = json.loads(json_txt)

                    list_actions = json_data['listActions']
                    words = get_bow_descriptors(list_actions)

                    res = instance_nn.predict_model_fast(words)

                    obj_to_write = {
                        'class': int(res['classes']),
                        'probabilities': res['probabilities'].tolist(),
                        'modelAccuracy': accuracy
                    }

                    obj_txt = json.dumps(obj_to_write, indent=4)
                    new_filename = ClassUtils.change_ext_training(full_path, '{0}_ensembledata'.format(option.value))

                    with open(new_filename, 'w') as f:
                        f.write(obj_txt)

                    # Done Writing

    print('Done!')


def get_bow_descriptors(list_actions):
    words = [0 for _ in range(12)]
    for action in list_actions:
        count = action['count']
        cls = action['class']

        for _ in range(count):
            words[cls] += 1

    return np.asanyarray(words, np.float)


if __name__ == '__main__':
    main()

