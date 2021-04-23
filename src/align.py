from __future__ import division
import numpy as np
import SimpleITK as sitk
from _mutinfo import mutual_information


def offset_slice(pixels1, pixels2, i, j):
    '''Return two sliced arrays where the first slice is offset by i,j
    relative to the second slice.
    '''
    if i < 0:
        height = min(pixels1.shape[0] + i, pixels2.shape[0])
        p1_imin = -i
        p2_imin = 0
    else:
        height = min(pixels1.shape[0], pixels2.shape[0] - i)
        p1_imin = 0
        p2_imin = i
    p1_imax = p1_imin + height
    p2_imax = p2_imin + height
    if j < 0:
        width = min(pixels1.shape[1] + j, pixels2.shape[1])
        p1_jmin = -j
        p2_jmin = 0
    else:
        width = min(pixels1.shape[1], pixels2.shape[1] - j)
        p1_jmin = 0
        p2_jmin = j
    p1_jmax = p1_jmin + width
    p2_jmax = p2_jmin + width

    p1 = pixels1[p1_imin:p1_imax, p1_jmin:p1_jmax]
    p2 = pixels2[p2_imin:p2_imax, p2_jmin:p2_jmax]
    return p1, p2


def sitk_translation(img0, img1, off0=0, off1=0):
    s0, s1 = sitk.GetImageFromArray(img0), sitk.GetImageFromArray(img1)
    s0, s1 = sitk.Cast(s0, sitk.sitkFloat32), sitk.Cast(s1, sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=250)
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)

    cc = sitk.TranslationTransform(s0.GetDimension())
    cc.SetOffset([-off1, -off0])
    R.SetInitialTransform(cc)
    R.SetInterpolator(sitk.sitkLinear)

    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([8, 4, 1])

    outTx = R.Execute(s0, s1)
    return -int(round(outTx.GetParameters()[1])), -int(round(outTx.GetParameters()[0]))


def register_multiseeds(img0, img1, bins=250, seeds=(-30, 0, 30)):
    store = []
    assert img0.shape == img1.shape
    for i in seeds:
        for ii in seeds:
            s0, s1 = sitk_translation(img0, img1, i, ii)
            p2, p1 = offset_slice(img1, img0, s0, s1)
            store.append(((s0, s1), mutual_information(p1, p2, bins)))
    store.sort(key=lambda x: x[1])
    return store[-1]



def calc_crop_coordinates(store, shapes):
    """
    Calculate how each images should be cropped.
    Take output from calc_jitters_multiple and return (start, end) coordinates for x, y for each image.
    """
    max_w = max(i[1] for i in store)
    start_w = [max_w - i[1] for i in store]
    size_w = min([shapes[1] + i[1] for i in store]) - max_w
    max_h = max(i[0] for i in store)
    start_h = [max_h - i[0] for i in store]
    size_h = min([shapes[0] + i[0] for i in store]) - max_h
    return [(hi, hi+size_h, wi, wi+size_w) for hi, wi in zip(start_h, start_w)]



def calc_jitters_mutual(images, seeds=(-20, -10, 0, 10, 20)):
    """
    Take multiple images and calculate jitters for consecutive images.
    The calculated values are based on the coordinates of the first image.
    """
    store = [(0, 0)]
    mhstore = []

    for i in range(1, images.shape[0]):
        img0, img1 = images[0], images[i]
        # crop outside
        c0, c1 = int(img0.shape[0] * 0.1), int(img0.shape[1] * 0.1)
        img0, img1 = img0[c0:-c0, c1:-c1], img1[c0:-c0, c1:-c1]

        jit, mh = register_multiseeds(img0, img1, seeds=seeds)
        store.append(jit)
        mhstore.append(mh)
    return store, mhstore


def calc_crop(images, info_thres=0.2):
    store, mh = calc_jitters_mutual(images)
    jitters = calc_crop_coordinates(store, images[0].shape)
    return jitters

