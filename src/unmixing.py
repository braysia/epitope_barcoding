import numpy as np
from scipy.optimize import nnls
from utils import ImageDict


def unmixing(field, weight):
    corr_field = ImageDict()
    stacked = np.vstack([i.ravel() for i in field.itervalues()])
    corrected = np.dot(np.linalg.inv(weight), stacked)
    for num, key in enumerate(field.iterkeys()):
        vec = corrected[num, :]
        corr_field[key] = vec.reshape(field[key].shape)
    return corr_field


def unmixing_nnls(field, weight):
    corr_field = ImageDict()
    stacked = np.vstack([i.ravel() for i in field.itervalues()])
    corrected = np.zeros(stacked.shape)
    for vec in range(stacked.shape[1]):
        corrected[:, vec] = nnls(weight, stacked[:, vec])[0]
    for num, key in enumerate(field.iterkeys()):
        vec = corrected[num, :]
        corr_field[key] = vec.reshape(field[key].shape)
    return corr_field

