
from sklearn import preprocessing as ps
import numpy as np


def min_max_scale(x):
    return ps.minmax_scale(x)


def z_score_scale(x):
    return ps.scale(x)


def shift_percentile(x):
    percentile = np.ceil(np.log10(np.max(np.abs(x))))
    return x / (10 ** percentile)


if __name__ == '__main__':
    '''
    min max scale
    z score scale
    shift percentile scale
    '''
    x = np.array([[0, -3, 1],[3, 1, 2],[0, 1, -1]])
    print(min_max_scale(x))
    print(z_score_scale(x))
    print(shift_percentile(x))
    '''
    home work
    '''
    x = np.array([5000,16000,58000])
    print(ps.minmax_scale(x))