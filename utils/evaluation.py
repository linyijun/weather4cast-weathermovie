import numpy as np
import math
from sklearn.metrics import r2_score


def compute_error(ground_truth, prediction, metrics=['rmse', 'r2']):
    errors = dict()
    if 'rmse' in metrics:
        errors['rmse'] = rmse_error(ground_truth, prediction)
    if 'r2' in metrics:
        errors['r2'] = r2_error(ground_truth, prediction)
    return rmse, r2


def r2_error(ground_truth, prediction):
    return r2_score(ground_truth[~np.isnan(ground_truth)], prediction[~np.isnan(ground_truth)])


def rmse_error(ground_truth, prediction):
    mse = np.nanmean((ground_truth - prediction) ** 2)
    return math.sqrt(mse)

