"""
Deprecated
Only used to train one vs all method
For training SVM - Use:
posesvmtraining
"""

from classloaddescriptors import ClassLoadDescriptors, EnumDesc
import numpy as np
import time
from sklearn import svm
from sklearn.externals import joblib
import math
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

model_dir = '/home/mauricio/models/svm_pose'


def main():
    Tk().withdraw()

    print('Initializing main function')

    # Checking model dir directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    res = input('Press 1 to get all points - 2 to get points transformed: ')

    if res == '1':
        type_desc = EnumDesc.ALL_TRANSFORMED
        load_descriptors(type_desc)
    elif res == '2':
        type_desc = EnumDesc.POINTS
        load_descriptors(type_desc)
    else:
        raise Exception('Option not implemented')


def load_descriptors(type_desc: EnumDesc):
    print('Loading descriptors using type: {0}'.format(type_desc))

    # Generating elements into list
    results = ClassLoadDescriptors.load_pose_descriptors(type_desc)

    training_data = results['trainingData']
    training_labels = results['trainingLabels']
    eval_data = results['evalData']
    eval_labels = results['evalLabels']
    classes_number = results['classesNumber']

    res = input('Press 1 to train, press 2 to eval, press 3 to eval file: ')

    if res == '1':
        train_model(training_data, training_labels, eval_data, eval_labels, classes_number)
    elif res == '2':
        eval_model(eval_data, eval_labels, classes_number)
    elif res == '3':
        eval_model_file(type_desc, classes_number)
    else:
        raise Exception('Option not implemented')


def train_model(train_data: np.ndarray, train_labels: np.ndarray,
                eval_data: np.ndarray, eval_labels: np.ndarray,
                classes_number: int):
    print('Initializing training data')

    # Now we have to train SVM model!
    print('Training MNIST model one vs all')

    data = train_data[0]
    print('Data: {0}'.format(data))

    list_models = list()

    for cls in range(classes_number):
        list_labels = list()

        for i in range(train_labels.shape[0]):
            if cls == int(train_labels[i]):
                list_labels.append(1)
            else:
                list_labels.append(0)

        new_labels_np = np.array(list_labels, dtype=np.int)

        print('Generating values to train in SVM: {0}'.format(cls))

        start_time = time.time()
        clf = svm.SVC(gamma=0.1, C=1000, probability=True)
        clf.fit(train_data, new_labels_np)
        end_time = time.time()

        print('Total time: {0}'.format((end_time - start_time)))

        path_model = os.path.join(model_dir, 'model{0}.pkl'.format(cls))
        print('Generating model into: {0}'.format(path_model))

        joblib.dump(clf, path_model)
        list_models.append(clf)

    eval_model(eval_data, eval_labels, classes_number)

    print('Done!')


def eval_model(eval_data: np.ndarray, eval_labels: np.ndarray, classes_number: int):
    print('Initializing eval data')

    # Reloading pose classification SVM
    print('Evaluating SVN multi class')

    print('Loading models')
    list_models = list()

    for cls in range(classes_number):
        path_model = os.path.join(model_dir, 'model{0}.pkl'.format(cls))

        print('Loading model {0}'.format(cls))
        model = joblib.load(path_model)
        list_models.append(model)

    data_ok = 0
    total_data = eval_labels.shape[0]
    for index in range(total_data):
        data = eval_data[index]
        array = np.expand_dims(data, axis=0)

        selected_cls = -1
        max_prob = -math.inf

        for cls in range(classes_number):
            model = list_models[cls]
            res = model.predict(array)
            res_prob = model.predict_log_proba(array)

            if res[0] == 1:
                print(res_prob)
                prob = res_prob[0][res[0]]

                if prob > max_prob:
                    selected_cls = cls
                    max_prob = prob

        if selected_cls == int(eval_labels[index]):
            data_ok += 1

    print('Data ok: {0}'.format(data_ok))
    print('Total data: {0}'.format(total_data))

    precision = data_ok / total_data
    print('Precision: {0}'.format(precision))

    print('Done!')


def eval_model_file(type_desc: EnumDesc, classes_number):
    init_dir = '/home/mauricio/Pictures/PosesNew'
    options = {'initialdir': init_dir}
    filename = askopenfilename(**options)

    if filename is None:
        raise Exception('Filename not selected!!!')

    desc = ClassLoadDescriptors.load_file_descriptors(filename, type_desc)
    array = np.expand_dims(desc, axis=0)

    print('Loading models')
    list_models = list()

    for cls in range(classes_number):
        path_model = os.path.join(model_dir, 'model{0}.pkl'.format(cls))

        print('Loading model {0}'.format(cls))
        model = joblib.load(path_model)
        list_models.append(model)

    selected_cls = -1
    max_prob = -math.inf

    for cls in range(classes_number):
        model = list_models[cls]
        res = model.predict(array)
        res_prob = model.predict_log_proba(array)

        if res[0] == 1:
            print(res_prob)
            prob = res_prob[0][res[0]]

            if prob > max_prob:
                selected_cls = cls
                max_prob = prob

    print('Selected cls: {0}'.format(selected_cls))
    print('Max prob: {0}'.format(max_prob))
    print('Done!')


if __name__ == '__main__':
    main()



