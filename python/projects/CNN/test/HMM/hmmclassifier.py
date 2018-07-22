"""
Testing -> Training Hidden Markov Models for classification

"""
import numpy as np
from hmmlearn import hmm


def main():
    print('Init main function')
    seq1 = np.array([[3, 2, 3, 2, 3], [1, 2, 1, 2, 2], [0, 1, 0, 1, 0]])
    seq2 = np.array([[2, 2, 2, 1, 1], [1, 1, 1, 0, 0], [3, 3, 3, 2, 2]])
    len = [3]

    hidden_states = 5
    model1 = hmm.MultinomialHMM(n_components=hidden_states, n_iter=1000)
    model2 = hmm.MultinomialHMM(n_components=hidden_states, n_iter=1000)

    model1.fit(seq1, len)
    model2.fit(seq2, len)

    seqTest = np.array([[3, 2, 3, 2, 3]])
    prob1 = model1.score(seqTest)
    prob2 = model2.score(seqTest)

    print(prob1)
    print(prob2)

    seqTest2 = np.array([[2, 2, 2, 1, 1]])
    prob1 = model1.score(seqTest2)
    prob2 = model2.score(seqTest2)

    print(prob1)
    print(prob2)
    print('Done!')


if __name__ == '__main__':
    main()
