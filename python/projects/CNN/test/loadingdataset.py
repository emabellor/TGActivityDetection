from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from classimagedataset import ClassImageDataSet

positive_dir = '/home/mauricio/Datasets/Example/Pos'
negative_dir = '/home/mauricio/Datasets/Example/Neg'


def main():
    print('Loading dataset')
    instance = ClassImageDataSet(positive_dir, negative_dir)

    pos_train, neg_train = instance.load_train_set()
    pos_eval, neg_eval = instance.load_eval_set()

    print('Test loading mnist dataset')
    pos_mnist_train, pos_mnist_eval = instance.load_eval_mnist()

    print('Train dataset')
    print(pos_mnist_train[0])

    print('Done!')


if __name__ == '__main__':
    main()
