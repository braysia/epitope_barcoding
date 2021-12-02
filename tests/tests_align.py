import os
import sys
module_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(module_dir, 'src'))
import numpy as np
from utils import ImageDict
from imageio import imread

def synthetic_data():
    img = imread('data/sample.png')
    img0 = img[:-20, :-20]
    img1 = img[20:, 20:]
    img2 = img[10:-10, 10:-10]
    images = np.stack([img0, img1, img2], axis=0)
    return images


def tests_align():
    images = synthetic_data()
    from align import calc_crop
    jitters = calc_crop(images)
    aligned = [im[j[0]:j[1], j[2]:j[3]] for j, im in zip(jitters, images)]
    np.testing.assert_array_equal(aligned[0], aligned[1])
    np.testing.assert_array_equal(aligned[1], aligned[2])
