"""
Using 49 components
As described in this link
https://www.kaggle.com/ddmngml/pca-and-svm-on-mnist-dataset
"""

import numpy as np
from classnn import ClassNN
from classimagedataset import ClassImageDataSet
from sklearn.decomposition import PCA


def main():
    print('Initializing main function')
    print('Loading datasets')

    train_data, train_labels = ClassImageDataSet.load_train_mnist()
    eval_data, eval_labels = ClassImageDataSet.load_eval_mnist()

    print('PCA with training data')
    n_features = 18

    pca = PCA(n_components=n_features, svd_solver='randomized').fit(train_data)
    train_pca = pca.transform(train_data)
    n_classes = 10
    eval_pca = pca.transform(eval_data)

    print('Printing shapes')
    print(train_data.shape)
    print(train_pca.shape)
    model_dir = '/tmp/model_example_pca'
    classifier = ClassNN(model_dir, n_classes)

    var = input('Set 1 to train, 2 to predict. Otherwise to eval ')

    if var == '1':
        print('Training model')
        classifier.train_model(train_pca, train_labels)
    elif var == '2':
        print('Predict model')
        print('Total elements: ' + str(eval_pca.shape[0]))
        index = 1100
        eval_item = eval_pca[index]
        print(eval_item.shape)

        result = classifier.predict_model(eval_item)
        print('Result obtained: ' + str(result['classes']))
        print('Print probabilities')
        print(result['probabilities'])

        print('Real result: ' + str(eval_labels[index]))
    else:
        print('Evaluating model')
        classifier.eval_model(eval_pca, eval_labels)

    print('Done!')


if __name__ == '__main__':
    main()

