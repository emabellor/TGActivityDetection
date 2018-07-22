"""
Implementation of multinomial HMM
Please refer to this link
http://hmmlearn.readthedocs.io/en/stable/api.html#multinomialhmm

The example on stock market is referred in the following link:
https://codefying.com/2016/09/15/a-tutorial-on-hidden-markov-model-with-a-stock-price-example/
"""
import numpy as np
from hmmlearn import hmm
import math

def main():
    print('Init main function')
    print('Creating model')

    model = hmm.MultinomialHMM(n_components=2)
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.7, 0.3], [0.42, 0.58]])
    model.emissionprob_ = np.array([[0.8, 0.15, 0.05], [0.25, 0.65, 0.1]])

    len = [1]
    array = [[0, 0, 0]]
    score_log = model.score(array, len)

    score = math.exp(score_log)

    print('Done')
    print(score)


if __name__ == '__main__':
    main()
