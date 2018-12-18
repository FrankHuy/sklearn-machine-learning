import numpy as np
from math import sqrt
from collections import Counter


def kNN_Classify(k, X_train, Y_train, x):

    assert 1 <= k <= X_train.shape[0], "K must be valid"
    assert X_train.shape[0] == Y_train.shape[0],\
        "The size of X_train must be equal to the size of Y_train"
    assert X_train.shape[1] == x.shape[0],\
        "The feature number of x must be equal to X_train"
    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    neareast = np.argsort(distances)
    topK_y = [Y_train[i] for i in neareast[:k]]
    statistic = Counter(topK_y)
    return statistic.most_common(1)[0][0]
