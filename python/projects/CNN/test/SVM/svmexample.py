"""
SVM example taken from this link
https://www.quantstart.com/articles/Support-Vector-Machines-A-Guide-for-Beginners
"""
import numpy as np
from sklearn import svm
from classimagedataset import ClassImageDataSet
from sklearn.externals import joblib


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


def main():
    option = input('Select 0 to train model, 1 to evaluate: ')

    if option == '0':
        train_model()
    else:
        eval_model()

    print('Program done!')


if __name__ == '__main__':
    main()
