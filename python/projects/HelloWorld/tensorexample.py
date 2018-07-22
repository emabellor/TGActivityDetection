import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']


def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL"""

    # Read train path from keras
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1], origin=TRAIN_URL)

    # Read train path datagram
    train = pd.read_csv(filepath_or_buffer=train_path, names=CSV_COLUMN_NAMES, header=0)

    # Read train features
    train_features, train_label = train, train.pop(label_name)

    # Read test data, as train data
    test_path = tf.keras.utils.get_file(fname=TEST_URL.split('/')[-1], origin=TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    return (train_features, train_label), (test_features, test_label)


def train_input_fn(features, labels, batch_size):
    """Function trainning - Neural Network"""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))  # type: tf.data.Dataset
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)  # type: tf.data.Dataset

    assert batch_size is not None, 'batch size must not be none'
    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def main():
    """Main function"""

    (train_x, train_y), (test_x, test_y) = load_data()

    # Create features description
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Define model - two hidden layers
    classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[10, 10],
                                            n_classes=3)

    batch_size = 100
    train_steps = 1000

    print('Training...')
    classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, batch_size), steps=train_steps)

    print('Training finalized. Testing...')
    eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_x, test_y, batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()


