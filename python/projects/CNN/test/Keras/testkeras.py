import tensorflow as tf
from tensorflow import keras
import numpy as np


def main():
    print('Initializing main function')
    test_nn()


def test_nn():
    print('Test fully connected ')

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    inputs = keras.Input(shape=(32,))  # Returns a placeholder tensor

    # A layer instance is callable on a tensor, and returns a tensor.
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    predictions = keras.layers.Dense(10, activation='softmax')(x)

    # Instantiate the model given inputs and outputs.
    model = keras.Model(inputs=inputs, outputs=predictions)

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Trains for 5 epochs
    model.fit(data, labels, batch_size=32, epochs=5)

    # Done!
    print('Done')


if __name__ == '__main__':
    main()
