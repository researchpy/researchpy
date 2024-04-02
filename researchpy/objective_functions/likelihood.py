import numpy as np

def log_likelihood(y_e):

    return np.sum(y_e - np.sum(np.log((1 + y_e))))

