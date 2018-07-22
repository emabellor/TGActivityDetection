"""
PCA variance analyzer
Tutorial from link: https://www.kaggle.com/ddmngml/pca-and-svm-on-mnist-dataset
"""

import matplotlib.pyplot as plt
from classimagedataset import ClassImageDataSet
from sklearn.decomposition import PCA


def main():
    print('Initializing main function')
    train_data, train_labels = ClassImageDataSet.load_train_mnist()
    components = 48
    pca = PCA(n_components=components, svd_solver='randomized').fit(train_data)  # type: PCA

    train_pca = pca.transform(train_data)
    print(pca.explained_variance_ratio_.sum())
    plt.hist(pca.explained_variance_ratio_, bins=components, log=True)
    plt.show()

    print('Done!')


if __name__ == '__main__':
    main()

