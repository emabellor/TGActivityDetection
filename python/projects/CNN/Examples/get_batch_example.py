import numpy as np


def main():
    print('Getting batch example')
    print('Creating numpy array')

    train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
    eval_data = np.array([[1], [2], [3], [4], [5], [6]])

    samples = train_data.shape[0]
    print('Samples: ' + str(samples))
    batch_size = 2
    number_samples = int(samples / batch_size)
    print('Number of samples: ' + str(number_samples))

    for i in range(number_samples):
        batch_x, batch_y = get_batch(train_data, eval_data, batch_size, i)
        print('Getting batches for ' + str(i))
        print(batch_x)
        print(batch_y)


def get_batch(train_data: np.ndarray, train_output: np.ndarray, batch_size: int, index: int):
    index_array = batch_size * index
    return train_data[index_array:batch_size+index_array, :], train_output[index_array:batch_size+index_array, :]


if __name__ == "__main__":
    main()
