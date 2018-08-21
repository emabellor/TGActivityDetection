# Multinomial HMM
# Scikit learn
# Documentation of the HMM module view link
# http://hmmlearn.readthedocs.io/en/latest/tutorial.html#saving-and-loading-hmm

import numpy as np
from hmmlearn import hmm
import os
from sklearn.externals import joblib
from classutils import ClassUtils
import json


class ClassHMM:
    def __init__(self, model_path: str):
        self.model = None  # type: hmm.MultinomialHMM

        ext = ClassUtils.get_filename_extension(model_path)

        if ext != '.pkl':
            raise Exception('Extension of model must be .pkl')

        self.model_path = model_path
        self.list_mapping = []

        # Trying to load model
        if os.path.exists(model_path):
            print('Loading model from {0}'.format(model_path))
            self.model = joblib.load(model_path)

            map_path = model_path.replace('.pkl', '.json')
            print('Loading list_mapping from {0}'.format(map_path))

            with open(map_path, 'r') as file:
                map_str = file.read()
                self.list_mapping = json.loads(map_str)
        else:
            print('Model {0} must be trained'.format(self.model_path))

    def train(self, train_data: list, hidden_states: int):
        self.model = hmm.MultinomialHMM(n_components=hidden_states, n_iter=1000)

        # Generating mapped list
        self._get_list_mapping(train_data)

        # Generate mapped model
        new_train_data = train_data.copy()

        for row in range(len(new_train_data)):
            for col in range(len(new_train_data[row])):
                new_train_data[row][col] = self._get_mapped_data(train_data[row][col])

        # Prepare data
        list_data_np = None
        lengths = list()
        for data in new_train_data:
            data_np = np.asarray(data, np.int)
            data_np = data_np[:, np.newaxis]

            if list_data_np is None:
                list_data_np = data_np
            else:
                list_data_np = np.concatenate([list_data_np, data_np])

            lengths.append(len(data))

        # Numpy array
        self.model.fit(list_data_np, lengths)

        # Saving model
        dir_name = os.path.dirname(self.model_path)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        joblib.dump(self.model, self.model_path)

        # Saving mapping
        map_str = json.dumps(self.list_mapping)
        map_path = self.model_path.replace('.pkl', '.json')
        with open(map_path, 'w') as file:
            file.write(map_str)

        # Done

    def _get_list_mapping(self, list_data):
        list_mapping = list()
        state_hmm = 0

        for data in list_data:
            for elem in data:

                found = False
                for data_map in list_mapping:
                    if data_map[0] == elem:
                        found = True
                        break

                if not found:
                    list_mapping.append((int(elem), state_hmm))
                    state_hmm += 1

        self.list_mapping = list_mapping
        # Done - return void

    def _get_mapped_data(self, data):
        for map_data in self.list_mapping:
            if map_data[0] == data:
                return map_data[1]

        # If not returned
        raise Exception('Cant find mapped for data {0}'.format(data))

    def _is_mapped(self, data):
        mapped = False

        for map_data in self.list_mapping:
            if map_data[0] == data:
                mapped = True

        return mapped

    def get_score(self, eval_data: list):
        # Only one dimension
        if self.model is None:
            raise Exception('Model not trained')
        else:
            # Check mapped data
            # If there are a not mapped state, returns a very low probability
            # -1000

            all_mapped = True
            for data in eval_data:
                if not self._is_mapped(data):
                    all_mapped = False
                    break

            if not all_mapped:
                print('States not mapped in model - Returns Inf probability')
                return -float('Inf')
            else:
                # Prepare data
                new_eval_data = eval_data.copy()
                for index in range(len(new_eval_data)):
                    new_eval_data[index] = self._get_mapped_data(eval_data[index])

                new_eval_data_np = np.asarray(new_eval_data, dtype=np.int)
                new_eval_data_np = new_eval_data_np[:, np.newaxis]
                len_data = [len(new_eval_data)]

                score = self.model.score(new_eval_data_np, len_data)
                return score  # Log based
