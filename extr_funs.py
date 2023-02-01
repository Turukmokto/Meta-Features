import numpy as np


def entropy(Y):
    unique, count = np.unique(Y, return_counts=True)
    prob = count / len(Y)
    en = np.sum((-1) * prob * np.log(prob))
    return en