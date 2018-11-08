from classutils import ClassUtils
from classnn import ClassNN
import os


list_classes = [
    {
        # Cls 0
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Door'),
    },
    {
        # Cls1
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Down'),
    },
    {
        # Cls 2
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Loitering'),
    },
    {
        # Cls 3
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Plumbs'),
    },
    {
        # Cls 4
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Squat'),
    },
    {
        # Cls 4
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Up'),
    },
    {
        # Cls 4
        'folderPath': os.path.join(ClassUtils.activity_base_path, 'Walk'),
    }
]


def main():
    print('Initializing Main Function')

    res = input('Press 1 to train using NN')
    if res == '1':
        calculate_poses_nn()


def calculate_poses_nn():
    print('Calculating poses using nn')

    classes_number = 10
    hidden_number = 60
    learning_rate = 0.04
    steps = 20000

    # Initialize classifier instance
    nn_classifier = ClassNN(model_dir=ClassNN.model_dir_pose,
                            classes=classes_number,
                            hidden_number=hidden_number,
                            learning_rate=learning_rate)


