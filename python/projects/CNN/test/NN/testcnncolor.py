import numpy as np
from classimagedataset import ClassImageDataSet
from classcnn import ClassCNN

positive_dir = '/home/mauricio/Datasets/Example/Pos'
negative_dir = '/home/mauricio/Datasets/Example/Neg'


def main():
    print('Running main function')

    # Loading images in color
    width_resize = 28
    height_resize = 28
    instance_load = ClassImageDataSet(positive_dir, negative_dir, width_resize, height_resize)

    train_data, train_labels = instance_load.load_train_set(load_color=True)
    eval_data, eval_labels = instance_load.load_eval_set(load_color=True)

    print('Printing shapes')
    print(train_data.shape)
    print(train_labels.shape)

    print('Generating instance')
    model_dir = '/tmp/cnn_color'
    n_classes = 2
    channels = 3

    classifier = ClassCNN(
        model_dir=model_dir,
        classes=n_classes,
        width=width_resize,
        height=height_resize,
        channels=channels,
        train_steps=5000
    )

    var = input('Set 1 to train, 2 to predict. Otherwise to eval ')

    if var == '1':
        print('Training model')
        classifier.train_model(train_data, train_labels)
    elif var == '2':
        print('Predict model')
        print('Total elements: ' + str(eval_data.shape[0]))
        index = 0
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

