from sklearn import svm
import numpy as np
from sklearn.externals import joblib
import os
from sys import platform


class ClassSVM:
    path_model_pose = '/home/mauricio/models/svm_classifier_pose'

    if platform == 'win32':
        path_model_pose = 'C:\\models\\svm_classifier_pose'

    def __init__(self, path_model):
        self.path_model = path_model

        if os.path.exists(self.path_model):
            print('Loading pre-trained model: {0}'.format(path_model))
            self.clf = joblib.load(path_model)
        else:
            self.clf = svm.SVC(gamma=0.1, C=1000, probability=True)

    def train_model(self, train_data_np: np.ndarray, train_labels_np: np.ndarray):
        # Training model
        # Creating new classifier
        self.clf = svm.SVC(gamma=0.1, C=1000, probability=True)
        self.clf.fit(train_data_np, train_labels_np)

        # Saving model using joblib library
        dir_path = os.path.dirname(self.path_model_pose)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        joblib.dump(self.clf, self.path_model)

    def eval_model(self, eval_data_np: np.ndarray, eval_labels_np: np.ndarray):
        score = self.clf.score(eval_data_np, eval_labels_np)
        return score

    def predict_model(self, data_np: np.ndarray):
        array_np = np.expand_dims(data_np, axis=0)
        cls = self.clf.predict(array_np)
        prob = self.clf.predict_proba(array_np)

        return {
            'classes': cls[0],
            'probabilities': prob[0]
        }
