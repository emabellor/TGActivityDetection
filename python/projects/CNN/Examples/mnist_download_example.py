import tensorflow as tf
import numpy as np

def main():
    print('Test downloading mnist')

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    print('Checking sizes')

    print('Size train data')
    print(train_data.shape)

    print('Size train labels')
    print(train_labels.shape)

    print('Printing label set')
    print(train_labels)


if __name__ == '__main__':
    main()
