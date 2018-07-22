"""
Implementation of pca functions used in this link
http://www.iro.umontreal.ca/~pift6080/H09/documents/papers/pca_tutorial.pdf
"""
import numpy as np


def main():
    print('Initializing main function')

    # 1 step -> Normalize data -> zero mean
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

    x_mean = np.mean(elems[:, 0])
    y_mean = np.mean(elems[:, 1])

    print(x_mean)
    print(y_mean)

    elems[:, 0] = elems[:, 0] - x_mean
    elems[:, 1] = elems[:, 1] - y_mean
    print(elems)

    # 2 step -> Getting covariance matrix
    cov_mat = np.cov(np.transpose(elems))
    print(cov_mat)

    # 3 step -> Calculate eigen values
    values, vectors = np.linalg.eig(cov_mat)
    print(values)
    print(vectors)

    # 4 step -> selecting eigenvector -> Hardcoded
    eigen_max = vectors[:, 1]

    # 5 step -> Getting values
    values = np.matmul(np.transpose(eigen_max), np.transpose(elems))
    print(values)

    print('Done getting PCA Features reduction')


if __name__ == '__main__':
    main()

