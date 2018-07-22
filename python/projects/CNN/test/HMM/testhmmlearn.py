import numpy as np
from hmmlearn import hmm


def main():
    np.random.seed(42)

    print('Working making predictions')
    print('Creating sequences')

    X1 = [[0.5], [1.0], [-1.0], [0.42], [0.24]]
    X2 = [[2.4], [4.2], [0.5], [-0.24]]

    print('Intializing matrix training')
    X = np.concatenate([X1, X2])
    lengths = [len(X1), len(X2)]

    model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=1000)
    model.fit(X, lengths)

    Z2 = model.predict(X1)
    print('Done predicting elements')
    print(Z2)


if __name__ == '__main__':
    main()
