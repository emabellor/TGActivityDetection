"""
SVM example taken from this link
https://www.quantstart.com/articles/Support-Vector-Machines-A-Guide-for-Beginners
"""
import numpy as np
from sklearn import svm
from classimagedataset import ClassImageDataSet
from sklearn.externals import joblib
import time


def eval_model():
    print('Evaluating MNIST model')
    print('Loading model')
    clf = joblib.load('/tmp/model.pkl')  # type: svm.SVC

    print('Loading train data')
    test, labels_test = ClassImageDataSet.load_eval_mnist()

    elem = test[0]

    print('Evaluating model in elements')
    elems = 0
    pos = 0
    for elem in test:
        array = np.expand_dims(elem, axis=0)

        prediction = clf.predict(array)

        if prediction[0] == labels_test[elems]:
            pos = pos + 1

        elems = elems + 1
        if elems % 10 == 0:
            print('Elems: ' + str(elems) + ' pos: ' + str(pos) + ' total: ' + str(labels_test.shape))

    precision = pos / elems

    print('Pos: ' + str(pos))
    print('Elems: ' + str(elems))
    print('Precision: ' + str(precision))
    print('Done!')


def train_model():
    print('Training MNIST model')

    train, labels = ClassImageDataSet.load_train_mnist()

    print('Generating values to train in SVM')
    clf = svm.SVC(gamma=0.001, C=100.)

    clf.fit(train, labels)
    print('Saving model in temp folder')
    joblib.dump(clf, '/tmp/model.pkl')


def train_model_one_vs_all():
    print('Training MNIST model one vs all')

    train_data, train_labels = ClassImageDataSet.load_train_mnist()
    data = train_data[0]
    print('Data: {0}'.format(data))
    classes = 10

    list_models = list()

    for cls in range(classes):
        list_labels = list()

        for i in range(train_labels.shape[0]):
            if cls == int(train_labels[i]):
                list_labels.append(1)
            else:
                list_labels.append(0)

        new_labels_np = np.array(list_labels, dtype=np.int)

        print('Generating values to train in SVM: {0}'.format(cls))

        start_time = time.time()
        clf = svm.SVC(gamma=0.001, C=100.)
        clf.fit(train_data, new_labels_np)
        end_time = time.time()

        print('Total time: {0}'.format((end_time - start_time)))

        path_model = '/tmp/model{0}.pkl'.format(cls)
        print('Generating model into: {0}'.format(path_model))

        joblib.dump(clf, path_model)
        list_models.append(clf)

    print('Evaluating model list')
    eval_data, eval_labels = ClassImageDataSet.load_eval_mnist()

    print('Getting first data!')
    print('Label: {0}'.format(eval_labels[0]))

    for index, model in enumerate(list_models):
        array = np.expand_dims(eval_data[0], axis=0)
        result = model.predict(array)

        print('Result for class {0}: {1}'.format(index, result))

    print('Done!')


def eval_model_one_vs_all():
    print('Evaluating SVN multi class')

    print('Loading models')
    classes = 10
    list_models = list()

    for cls in range(classes):
        path_model = '/tmp/model{0}.pkl'.format(cls)

        print('Loading model {0}'.format(cls))
        model = joblib.load(path_model)
        list_models.append(model)

    # Loading eval classes
    eval_data, eval_labels = ClassImageDataSet.load_eval_mnist()

    data_ok = 0
    total_data = eval_labels.shape[0]
    for index in range(total_data):
        data = eval_data[index]
        array = np.expand_dims(data, axis=0)

        selected_cls = -1
        for cls in range(classes):
            model = list_models[cls]
            res = model.predict(array)

            if res[0] == 1:
                selected_cls = cls

        if selected_cls == int(eval_labels[index]):
            data_ok += 1

    print('Data ok: {0}'.format(data_ok))
    print('Total data: {0}'.format(total_data))

    precision = data_ok / total_data
    print('Precision: {0}'.format(precision))

    print('Done!')


def main():
    option = input('Select 1 to train model, 2 to evaluate, 3 to train one vs all, 4 to eval one vs all: ')

    if option == '1':
        train_model()
    elif option == '2':
        eval_model()
    elif option == '3':
        train_model_one_vs_all()
    elif option == '4':
        eval_model_one_vs_all()
    else:
        raise Exception('Option not implemented!')


if __name__ == '__main__':
    main()
