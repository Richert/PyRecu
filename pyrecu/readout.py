from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import numpy as np
import typing as tp


class ReadoutTraining:

    def __init__(self):

        self.readouts = {}
        self.readout_id = 0

    def get_coefs(self, key: tp.Any):
        return self.readouts[key].coef_

    def get_scores(self, key: tp.Any):
        return self.readouts[key].cv_values

    def get_classifier(self, key: tp.Any):
        return self.readouts[key]

    def ridge_fit(self, X: np.ndarray, y: np.ndarray, k: int = 1, verbose: bool = True, readout_key: tp.Any = None,
                  **kwargs):

        if readout_key is None:
            readout_key = self.readout_id
            self.readout_id += 1

        # perform ridge regression
        if k > 1:
            splitter = StratifiedKFold(n_splits=k)
            scores, coefs = [], []
            for i, (train_idx, test_idx) in enumerate(splitter.split(X=X, y=y)):
                classifier = Ridge(**kwargs)
                classifier.fit(X[train_idx], y[train_idx])
                scores.append(classifier.score(X=X[test_idx], y=y[test_idx]))
                coefs.append(classifier.coef_)
        else:
            classifier = Ridge(**kwargs)
            classifier.fit(X, y)
            scores = [classifier.score(X=X, y=y)]
            coefs = [classifier.coef_]

        # store readout weights
        w_out = np.mean(coefs, axis=0)
        self.readouts[readout_key] = w_out if len(w_out.shape) == 1 else w_out.T

        if verbose:
            print(f'Finished readout training. The readout weights are stored under the key: {readout_key}. '
                  f'Please use that key when calling `RNN.test()` or `RNN.predict()`.')
            avg_score = np.mean(scores)
            if k > 1:
                print(f'Average, cross-validated classification performance across {k} test folds: {avg_score}')
            else:
                print(f'Classification performance on training data: {avg_score}')

        return readout_key, scores, coefs

    def ridge_cv_fit(self, X: np.ndarray, y: np.ndarray, readout_key: tp.Any = None, verbose: bool = True, **kwargs):

        if readout_key is None:
            readout_key = self.readout_id
            self.readout_id += 1

        # perform ridge regression
        classifier = RidgeCV(**kwargs)
        classifier.fit(X, y)

        # store readout weights
        w_out = classifier.coef_
        self.readouts[readout_key] = w_out if len(w_out.shape) == 1 else w_out.T
        if verbose:
            print(f'Finished readout training. The readout weights are stored under the key: {readout_key}. '
                  f'Please use that key when calling `RNN.test()` or `RNN.predict()`.')

        return readout_key, classifier

    def test(self, X: np.ndarray, y: np.ndarray, readout_key: tp.Any = None):

        y_predict = self.predict(X, readout_key)
        return r2_score(y, y_predict), y_predict

    def predict(self, X: np.ndarray, readout_key):

        if readout_key is None:
            readout_key = self.readout_id

        return X @ self.readouts[readout_key]
