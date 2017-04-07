import pandas as pd
from itertools import product
import numpy as np
from os.path import basename, join, dirname
from operator import itemgetter


def pd_array_convert(path):
    df = pd.read_csv(path, index_col=['object', 'ch', 'prop', 'frame'])
    objects, channels, props = [list(i) for i in df.index.levels[:3]]
    labels = [i for i in product(objects, channels, props)]
    storage = []
    for i in labels:
        storage.append(np.float32(df.ix[i]).T)
    arr = np.rollaxis(np.dstack(storage), 2)

    dic_save = {}
    dic_save['data'] = arr
    dic_save['labels'] = labels

    # FILE NAME
    file_name = basename(path).split('.')[0]
    np.savez_compressed(join(dirname(path), file_name), **dic_save)


def save_output(arr, labels, time, path):
    dic_save = {'data': arr, 'labels': labels, 'time': time}
    np.savez_compressed(path, **dic_save)


def sort_labels_and_arr(labels, arr=[]):
    '''
    >>> labels = [['a', 'B', '1'], ['a', 'A', '1'], ['b', 'A', '3'], ['b', 'B', '2']]
    >>> sort_labels_and_arr(labels)
    [['a', 'A', '1'], ['a', 'B', '1'], ['b', 'A', '3'], ['b', 'B', '2']]
    >>> labels = [['a', 'B', '1'], ['prop'], ['aprop'], ['b', 'B', '2']]
    >>> sort_labels_and_arr(labels)
    [['a', 'B', '1'], ['aprop'], ['b', 'B', '2'], ['prop']]
    '''

    labels = [list(i) for i in labels]
    labels, sort_idx = sort_multi_lists(labels)
    if not len(arr):
        return labels
    if len(arr):
        if arr.ndim == 3:
            arr = arr[sort_idx, :, :]
        if arr.ndim == 2:
            arr = arr[sort_idx, :]
        return labels, arr


def uniform_list_length(labels):
    """
    Insert empty string untill all the elements in labels have the same length.

    Examples:

    >>> uniform_list_length([['a'], ['a', 'b'], ['a', 'b', 'c']])
    [['a', ' ', ' '], ['a', 'b', ' '], ['a', 'b', 'c']]
    """
    max_num = max([len(i) for i in labels])
    for label in labels:
        for num in range(1, max_num):
            if len(label) == num:
                label.extend([" " for i in range(max_num - num)])
    return labels


def undo_uniform_list_length(labels):
    """
    Remove empty string after the operation done by uniform_list_length.

    Examples:

    >>> undo_uniform_list_length(uniform_list_length([['a'], ['a', 'b'], ['a', 'b', 'c']]))
    [['a'], ['a', 'b'], ['a', 'b', 'c']]
    """
    for label in labels:
        while " " in label:
            label.remove(" ")
    return labels

def sort_multi_lists(labels):
    """
    Sort a list by the order of column 0, 1 and 2.
    Works for a list having different length of elements.
    Now only work for a list with two or three elements.

    Examples:
    >>> sort_multi_lists([['a', 'c'], ['a', 'b'], ['a', 'b', 'c']])
    ([['a', 'b'], ['a', 'b', 'c'], ['a', 'c']], [1, 2, 0])
    """
    unilabels = uniform_list_length(labels)
    intlist = [[i] * 3 for i in range(len(unilabels))]
    # sort_func = itemgetter(*range(len(unilabels[0])))
    try:
        sort_func = lambda item: (item[0][0], item[0][1], item[0][2])
        sort_idx = [ii[0] for (i, ii) in sorted(zip(unilabels, intlist), key=sort_func)]
    except:
        sort_func = lambda item: (item[0][0], item[0][1])
        sort_idx = [ii[0] for (i, ii) in sorted(zip(unilabels, intlist), key=sort_func)]
    sort_labels = [unilabels[i] for i in sort_idx]
    return undo_uniform_list_length(sort_labels), sort_idx
