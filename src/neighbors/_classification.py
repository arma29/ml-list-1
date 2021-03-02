from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from collections import Counter


class Knn(BaseEstimator, ClassifierMixin):
    """
    Custom Implementation of KNN using sklearn interfaces

    Estimator object
    """
    def __init__(self, n_neighbors=5, weights='uniform'):
        weights_lst = ['uniform', 'distance', 'adaptive']
        if(weights not in weights_lst):
            raise Exception(
                'The \'weights\' should be one of: uniform, distance, adaptive')
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.__is_fitted = False
        self.__y_pred_func = self.__get_y_pred_func()

    def fit(self, X, y):
        self.__fit_X = X
        self.__y = y
        self.__is_fitted = True

    def predict(self, X):
        if(not self.__is_fitted):
            raise Exception('Not fitted')
        training_examples = self.__create_training_examples(X)
        k_nearests = self.__get_k_nearests(training_examples)
        return self.__y_pred_func(k_nearests)

    def __create_training_examples(self, X):
        mtz_res = euclidean_distances(self.__fit_X, X)
        transposed = np.transpose(mtz_res)
        training_examples = [
            [[] for _ in range(len(transposed[0]))] for _ in range(len(transposed))]
        for i in range(len(transposed)):
            for j in range(len(transposed[i])):
                training_examples[i][j] = [transposed[i][j], self.__y[j]]
        return training_examples

    def __get_k_nearests(self, training_examples):
        return [sorted(x, key=lambda y: y[0])[:self.n_neighbors] for x in training_examples]

    def __get_y_pred_func(self):
        if(self.weights == 'uniform'):
            return self.__get_y_pred_uniform
        elif(self.weights == 'distance'):
            return self.__get_y_pred_distance

    def __get_y_pred_uniform(self, k_nearests):
        freqs = [[y[1] for y in x] for x in k_nearests]
        y_pred = [Counter(x).most_common(1)[0][0] for x in freqs]
        return y_pred

    def __delta(self, a,b):
        return 1 if a == b else 0

    def __get_y_pred_distance(self, k_nearests):
        V = np.unique(self.__y)
        v_arr = [[] for _ in range(len(k_nearests))]
        for v in V:
            for i in range(len(k_nearests)): # Foreach xq
                summ = 0
                for xk in k_nearests[i]: # Foreach k instance on xq
                    wi = 1/(xk[0]**2 + 1e-10)
                    summ += wi*self.__delta(v, xk[1])
                v_arr[i].append(summ)
                
        return [V[np.argmax(xq)] for xq in v_arr]

    def score(self, X, y):
        return self.__accuracy_score(y, self.predict(X))

    def __accuracy_score(self, y_true, y_pred):
        if(len(y_true) != len(y_pred)):
            raise Exception('Diff lens')
        hits = [x for x in np.subtract(y_true, y_pred) if x == 0]
        return len(hits)/len(y_true)
