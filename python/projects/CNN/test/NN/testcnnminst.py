"""
Testing nn class
"""
from classimagedataset import ClassImageDataSet
from classcnn import ClassCNN
import sys


def main():
    print('Init main function')
    print('Loading models')
    print('Load model in grayscale first')

    width = 28
    height = 28

    # Test as reshape image
    train_data, train_labels = ClassImageDataSet.load_train_mnist(reshape=True)
    eval_data, eval_labels = ClassImageDataSet.load_eval_mnist(reshape=True)

    n_classes = 10
    channels = 1

    model_dir = '/tmp/model_example_mnist'
    classifier = ClassCNN(
        model_dir=model_dir,
        channels=channels,
        classes=n_classes,
        width=width,
        height=height,
        train_steps=5000
    )

    var = input('Set 1 to train, 2 to predict. Otherwise to eval ')

    if var == '1':
        print('Training model')
        classifier.train_model(train_data, train_labels)
    elif var == '2':
        print('Predict model')
        print('Total elements: ' + str(eval_data.shape[0]))
        index = 1100
        eval_item = eval_data[index]
        print(eval_item.shape)

        result = classifier.predict_model(eval_item)
        print('Result obtained: ' + str(result['classes']))
        print('Print probabilities')
        print(result['probabilities'])

        print('Real result: ' + str(eval_labels[index]))
    else:
        print('Evaluating model')
        classifier.eval_model(eval_data, eval_labels)

    print('Done!')


if __name__ == '__main__':
    main()
