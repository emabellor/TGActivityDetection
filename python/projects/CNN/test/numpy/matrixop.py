import numpy as np


def main():
    print('Initializing main function')
    print('Generating elements in list')

    cov = np.array([[0.616555, 0.615444],
                  [0.615444, 0.716555]])

    eigen_A = np.array([-0.73517, 0.67787])
    eigen_B = np.array([[-0.67787], [-0.735178]])

    result1 = np.matmul(cov, eigen_A)
    result2 = np.matmul(cov, eigen_B)

    print('Result 1')
    print(result1)

    print('Result 2')
    print(result2)

    values, vectors = np.linalg.eig(cov)
    print('Result')
    print(values)
    print(vectors)

    test = eigen_A * 0.04908329
    print(test)

    print('Done!')


if __name__ == '__main__':
    main()
