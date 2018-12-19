import numpy as np
from math import sqrt
from collections import Counter

# def kNN_Classify(k, X_train, Y_train, x):

#     assert 1 <= k <= X_train.shape[0], "K must be valid"
#     assert X_train.shape[0] == Y_train.shape[0],\
#         "The size of X_train must be equal to the size of Y_train"
#     assert X_train.shape[1] == x.shape[0],\
#         "The feature number of x must be equal to X_train"
#     distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
#     neareast = np.argsort(distances)
#     topK_y = [Y_train[i] for i in neareast[:k]]
#     statistic = Counter(topK_y)
#     return statistic.most_common(1)[0][0]


class KNNClassifier:
    def __init__(self, k):
        assert k >= 1, "K must be valid"
        self.k = k
        self._X_train = None
        self._Y_train = None

    def fit(self, X_train, Y_train):
        assert X_train.shape[0] == Y_train.shape[0],\
         "The size of X_train must be equal to the size of Y_train"
        assert self.k <= X_train.shape[0],\
         "The size of X_train must be at least k"
        self._X_train = X_train
        self._Y_train = Y_train
        return self

    def predict(self, X_predict):
        assert self._X_train is not None and self._Y_train is not None,\
        "Must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1],\
        "The feature number of X_predict must be equal to X_train"
        Y_predict = [self._predict(x) for x in X_predict]
        return np.array(Y_predict)

    def _predict(self, x):
        assert self._X_train.shape[1] == x.shape[0],\
         "The feature number of x must be equal to X_train"
        distances = [
            sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train
        ]
        nearest = np.argsort(distances)
        topK_y = [self._Y_train[i] for i in nearest[:self.k]]
        vote = Counter(topK_y)
        return vote.most_common(1)[0][0]
