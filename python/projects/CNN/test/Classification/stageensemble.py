from classutils import ClassUtils
import os
import random
import json

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

    res = input('Press 1 to do ensemble classifier: ')
    if res == '1':
        do_ensemble()
    else:
        raise Exception('Option not implemented')


def do_ensemble():
    print('Doing ensemble')

    # Loading elements into list
    training_data = list()
    training_labels = list()

    eval_data = list()
    eval_labels = list()

    # Generating ensemble into list
    for index, item in enumerate(list_classes):
        folder = item['folderPath']

        list_data = list()

        for root, _, files in os.walk(folder):
            found = False
            votes = [0 for _ in range(len(list_classes))]
            for file in files:
                full_path = os.path.join(root, file)
                if '_1_ensembledata' in full_path:
                    found = True

                    with open(full_path, 'r') as f:
                        json_txt = f.read()

                    json_data = json.loads(json_txt)
                    cls = json_data['class']

                    votes[cls] += json_data['modelAccuracy']

            if found:
                list_data.append({
                    'root': root,
                    'votes': votes
                })

        # Split data using root folder for each prediction
        list_data.sort(key=lambda x: x['root'])

        # Use static seed to ensure same order
        random.Random(seed).shuffle(list_data)

        total_samples = len(list_data)
        total_train = int(total_samples * 80 / 100)
        print('Total samples: {0}'.format(total_samples))

        num_file = 0
        for data in list_data:
            if num_file < total_train:
                training_data.append(data)
                training_labels.append(index)
            else:
                eval_data.append(data)
                eval_labels.append(index)

            num_file += 1

    # Ensemble method
    # Apply only in eval_data
    counter = 0
    ok_files = 0

    for index in range(len(eval_data)):
        data = eval_data[index]
        label = eval_labels[index]

        m = max(data['votes'])
        max_pos_array = [i for i, j in enumerate(data['votes']) if j == m]
        max_pos = max_pos_array[0]

        if max_pos == label:
            ok_files += 1

        counter += 1

    precision = ok_files / counter
    print('Precision: {0}'.format(precision))

    print('Done!')


if __name__ == '__main__':
    main()

