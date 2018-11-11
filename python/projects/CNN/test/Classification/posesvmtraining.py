from classsvm import ClassSVM
from classloaddescriptors import ClassLoadDescriptors, EnumDesc

list_folder_data = [
    ('/home/mauricio/Pictures/PosesNew/Back', 0.05, 0),
    ('/home/mauricio/Pictures/PosesNew/Hands_Left', 0.05, 1),
    ('/home/mauricio/Pictures/PosesNew/Hands_Right', 0.05, 2),
    ('/home/mauricio/Pictures/PosesNew/Front', 0.05, 3),
    ('/home/mauricio/Pictures/PosesNew/Left', 0.05, 4),
    ('/home/mauricio/Pictures/PosesNew/Right', 0.05, 5),
    ('/home/mauricio/Pictures/PosesNew/Squat_Left', 0.05, 6),
    ('/home/mauricio/Pictures/PosesNew/Squat_Right', 0.05, 7),
    ('/home/mauricio/Pictures/PosesNew/Extend_Left', 0.05, 8),
    ('/home/mauricio/Pictures/PosesNew/Extend_Right', 0.05, 9)
]


def main():
    print('Initializing main function')

    res = input('Press 1 to train SVM for pose recognition - 2 to test prediction without training: ')

    if res == '1':
        train_svm_pose()
    elif res == '2':
        test_svm_prediction()
    else:
        raise Exception('Option not recognized: {0}'.format(res))


def train_svm_pose():
    print('Training SVM pose')

    # Loading elements
    results = ClassLoadDescriptors.load_pose_descriptors(EnumDesc.ALL)

    training_data_np = results['trainingData']
    training_labels_np = results['trainingLabels']
    eval_data_np = results['evalData']
    eval_labels_np = results['evalLabels']

    # Creating SVM classifier
    instance_svm = ClassSVM(ClassSVM.path_model_pose)

    # Training
    print('Training model')
    instance_svm.train_model(training_data_np, training_labels_np)

    # Evaluating
    print('Evaluating model')
    score = instance_svm.eval_model(eval_data_np, eval_labels_np)
    print('Score: {0}'.format(score))

    # Predict with probabilities
    print('Predicting with probabilities')
    res = instance_svm.predict_model(eval_data_np[-1])
    print(res)

    print('Done!')


def test_svm_prediction():
    print('Testing prediction')

    # Loading elements
    results = ClassLoadDescriptors.load_pose_descriptors(EnumDesc.ALL)

    training_data_np = results['trainingData']
    training_labels_np = results['trainingLabels']
    eval_data_np = results['evalData']
    eval_labels_np = results['evalLabels']

    # Creating SVM classifier
    instance_svm = ClassSVM(ClassSVM.path_model_pose)

    # Predict with probabilities
    print('Predicting with probabilities')
    res = instance_svm.predict_model(eval_data_np[-1])
    print(res)

    print('Done!')


if __name__ == '__main__':
    main()

