import numpy as np


def linear_cost(X, y, theta):
    m, _ = X.shape
    h = np.matmul(X, theta)
    sq = (h-y) ** 2
    return sq.sum() / (2 * m)
