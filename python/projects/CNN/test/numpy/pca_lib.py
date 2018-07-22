"""
Using PCA analysis by a library
Generating elements in list
"""

import numpy as np
from sklearn.decomposition import PCA


def main():
    print('Initializing main function')

    elems = np.array([[2.5, 2.4],
                      [0.5, 0.7],
                      [2.2, 2.9],
                      [1.9, 2.2],
                      [3.1, 3.0],
                      [2.3, 2.7],
                      [2, 1.6],
                      [1, 1.1],
                      [1.5, 1.6],
                      [1.1, 0.9]])

    pca = PCA(n_components=1, svd_solver='randomized').fit(elems)

    elems_pca = pca.transform(elems)

    print(elems_pca)
    print('Done!')


if __name__ == '__main__':
    main()

