import os
import sys
module_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(module_dir, 'src'))
import numpy as np
from utils import ImageDict

eps = 1e-6

def synthetic_data():
    img0 = np.random.random((100, 100))
    img1 = np.random.random((100, 100))
    raw, images = ImageDict(), ImageDict()
    raw['GFP'] = img0
    raw['RFP'] = img1

    images['GFP'] = img0 + img1 * 0.25
    images['RFP'] = img0 * 0.5 + img1
    weight = np.array([[1, 0.25], [0.5, 1]])
    return raw, images, weight

def tests_unmixing():
    from unmixing import unmixing
    raw, images, weight = synthetic_data()
    unmixed = unmixing(images, weight)
    subt = unmixed - raw
    assert sum([np.abs(i).sum() for i in subt.itervalues()]) < eps

def tests_unmixing_nnls():
    from unmixing import unmixing_nnls
    raw, images, weight = synthetic_data()
    unmixed = unmixing_nnls(images, weight)
    subt = unmixed - raw
    assert sum([np.abs(i).sum() for i in subt.itervalues()]) < eps
