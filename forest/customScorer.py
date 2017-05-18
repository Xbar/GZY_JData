# skewed data, cannot just use accuracy as evaluation score
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

def score_func(label, prediction, recall_precision_ratio=1.0):
    correct = np.sum(label * prediction) * 1.0
    precision = correct / (np.sum(prediction) + 1e-6) + 1e-6
    recall = correct / (np.sum(label) + 1e-6) + 1e-6
    return (1 + recall_precision_ratio) * recall * precision \
        / (recall_precision_ratio * recall + precision)
#    F11 = 6 * recall * precision / (5 * recall + precision)
#    F12 = 5 * recall * precision / (2 * recall + 3 * precision)
#    Score = 0.4 * F11 + 0.6 * F12
#    return Score

default_scorer = make_scorer(score_func, greater_is_better=True)