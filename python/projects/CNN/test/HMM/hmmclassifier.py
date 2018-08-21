"""
Testing -> Training Hidden Markov Models for classification

"""
import numpy as np
from hmmlearn import hmm
from classhmm import ClassHMM


def main():
    option = input('Select 1 to classic, 2 to class: ')

    if option == '1':
        classic()
    elif option == '2':
        class_method()
    else:
        print('Option not recognized: {0}'.format(option))


def classic():
    print('Init main function')
    seq1 = np.array([[3], [2], [3], [2], [3], [1], [2], [0], [1], [0], [1], [0]])
    seq2 = np.array([[0], [0], [0], [1], [1], [2], [2], [3], [3], [3], [2], [2]])
    seqTest = np.array([[3], [2], [3], [2], [3]])
    len = [5, 2, 5]

    hidden_states = 5
    model1 = hmm.MultinomialHMM(n_components=hidden_states, n_iter=1000)
    model1.fit(seq1, len)
    prob1 = model1.score(seqTest, [5])

    model2 = hmm.MultinomialHMM(n_components=hidden_states, n_iter=1000)
    model2.fit(seq2, len)
    prob2 = model2.score(seqTest, [5])

    print(prob1)
    print(prob2)

    seqTest2 = np.array([[0], [0], [0], [0], [0]])
    prob1 = model1.score(seqTest2, [5])
    prob2 = model2.score(seqTest2, [5])

    print(prob1)
    print(prob2)
    print('Done!')


def class_method():
    print('Class Method')

    seq1 = [[3, 2, 3, 2, 3], [1, 2], [0, 1, 0, 1, 0]]
    seq2 = [[3, 3, 3, 2, 2], [1, 1], [2, 2, 2, 3, 3]]

    model_1_path = '/home/mauricio/models/hmm/train1.pkl'
    model_2_path = '/home/mauricio/models/hmm/train2.pkl'

    model_1 = ClassHMM(model_1_path)
    model_2 = ClassHMM(model_2_path)

    hidden_states = 6
    model_1.train(seq1, hidden_states)
    model_2.train(seq2, hidden_states)

    seq_test = [3, 2, 3, 2, 3]
    score1 = model_1.get_score(seq_test)
    score2 = model_2.get_score(seq_test)
    print('Score 1: {0} - Score 2: {1}'.format(score1, score2))

    seq_test_2 = [4, 2, 3, 2, 3]
    score1 = model_1.get_score(seq_test_2)
    score2 = model_2.get_score(seq_test_2)
    print('Score 1: {0} - Score 2: {1}'.format(score1, score2))

    seq_test3 = [3, 3, 3, 3, 3]
    score1 = model_1.get_score(seq_test3)
    score2 = model_2.get_score(seq_test3)
    print('Score 1: {0} - Score 2: {1}'.format(score1, score2))

    print('Done!')


if __name__ == '__main__':
    main()
