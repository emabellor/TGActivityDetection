import time
from classnn import ClassNN
from classimagedataset import ClassImageDataSet
from classopenpose import ClassOpenPose
import numpy as np
from classdescriptors import ClassDescriptors
from classutils import ClassUtils

positive_dir = '/home/mauricio/Datasets/Example/Pos'
negative_dir = '/home/mauricio/Datasets/Example/Neg'

width_resize = 20
height_resize = 20


def main():
    print('Initializing main function')

    instance_train = ClassImageDataSet(positive_dir, negative_dir, width_resize, height_resize)

    train_data, train_labels = instance_train.load_train_mnist()
    eval_data, eval_labels = instance_train.load_eval_mnist()

    res = input('Press 1 to train, 2 to eval, 3 to eval fast, 4 to eval fast pose: ')
    if res == '1':
        train(train_data, train_labels)
    elif res == '2':
        evaluating(eval_data)
    elif res == '3':
        evaluating_fast(eval_data)
    elif res == '4':
        evaluating_fast_nn()
    else:
        raise Exception('Option not recognized!')


def train(train_data, train_labels):
    # Loading instances
    classes = 10
    hidden_layers = 50
    instance_nn = ClassNN(model_dir='/tmp/model', classes=classes, hidden_number=hidden_layers)

    print('Training...')
    instance_nn.train_model(train_data, train_labels)
    print('Done Training')


def evaluating(eval_data):
    # Loading instances
    classes = 10
    hidden_layers = 50
    instance_nn = ClassNN(model_dir='/tmp/model', classes=classes, hidden_number=hidden_layers)

    print('Evaluating...')

    data = instance_nn.predict_model(eval_data[0])
    print(data)

    data = instance_nn.predict_model(eval_data[1])
    print(data)

    start = time.time()
    data = instance_nn.predict_model(eval_data[2])
    end = time.time()

    print(data)
    print('Time elapsed: {0}'.format(end - start))

    print('Done evaluating')


def evaluating_fast(eval_data):
    print('Evaluating...')

    # Loading instances
    classes = 10
    hidden_layers = 50
    instance_nn = ClassNN(model_dir='/tmp/model', classes=classes, hidden_number=hidden_layers)

    data = instance_nn.predict_model_fast(eval_data[0])
    print(data)

    data = instance_nn.predict_model_fast(eval_data[1])
    print(data)

    start = time.time()
    data = instance_nn.predict_model_fast(eval_data[3])
    end = time.time()
    print(data)
    print('Time elapsed: {0}'.format(end - start))

    start = time.time()
    data = instance_nn.predict_model_fast(eval_data[0])
    end = time.time()

    print(data)
    print('Time elapsed: {0}'.format(end - start))

    print('Done evaluating fast')


def evaluating_fast_nn():
    print('Initializing evaluating fast nn')

    classes = 8
    hidden_layers = 40
    instance_nn = ClassNN(model_dir=ClassNN.model_dir_pose, classes=classes, hidden_number=hidden_layers)
    instance_pose = ClassOpenPose()

    info = ClassDescriptors.load_images_comparision_ext(instance_pose, min_score=0.05, load_one_img=True)
    pose1 = info['pose1']

    items = ClassDescriptors.get_person_descriptors(pose1, 0.05)

    # Valid pose for detection
    data_to_add = list()
    data_to_add += items['angles']
    data_to_add += ClassUtils.get_flat_list(items['transformedPoints'])

    data_np = np.asanyarray(data_to_add, dtype=np.float)
    result = instance_nn.predict_model_fast(data_np)
    print(result)
    key_pose = result['classes']
    probability = result['probabilities'][key_pose]

    print('Key pose: {0}'.format(key_pose))
    print('Probability: {0}'.format(probability))
    print('Done!')


if __name__ == '__main__':
    main()

