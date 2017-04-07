import numpy as np
from collections import OrderedDict
from datatype_handling import sort_labels_and_arr


class DArray(np.ndarray):
    """
    Each rows corresponds to labels, each columns corresponds to cells.
    >> arr = np.random.rand(3, 100)
    >> labelarr = np.array([['nuc' ,'area', ''], 
                             ['nuc' ,'YFP' , 'intensity'], 
                             ['nuc' ,'YFP' , 'intensity']], dtype=object)
    >> darr = DataArray(arr, labelarr)
    >> print darr['nuc', 'area].shape
    (1, 100)
    >> print darr['nuc', 'YFP', 'intensity'].shape
    (2, 100)
    """
    def __new__(cls, arr, labelarr=None):
        obj = np.asarray(arr).view(cls)
        obj.labelarr = labelarr
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.labelarr = getattr(obj, 'labelarr', None)

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self._label2idx(item)
        if isinstance(item, tuple):
            if isinstance(item[0], str):
                item = self._label2idx(item)
        self.item = item
        return super(DArray, self).__getitem__(item)

    def _label2idx(self, item):
        item = (item, ) if not isinstance(item, tuple) else item
        boolarr = np.ones(self.labelarr.shape[0], dtype=bool)
        for num, it in enumerate(item):
            boolarr = boolarr * (self.labelarr[:, num]==it)
        return np.where(boolarr)


class LabeledArray(DArray):
    def __new__(cls, arr=None, labelarr=None, idx=None):
        obj = np.asarray(arr).view(cls)
        obj.labelarr = labelarr
        obj.idx = idx
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.labelarr = getattr(obj, 'labelarr', None)
        if hasattr(obj, 'idx') and np.any(self.labelarr):
            self.labelarr = self.labelarr[obj.idx]
            if self.labelarr.ndim > 1:
                f_leftshift = lambda a1:all(x>=y for x, y in zip(a1, a1[1:]))
                all_column = np.all(self.labelarr == self.labelarr[0,:], axis=0)
                sl = 0 if not f_leftshift(all_column) else all_column.sum()
                self.labelarr = self.labelarr[:, slice(sl, None)]

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self._label2idx(item)
        if isinstance(item, tuple):
            if isinstance(item[0], str):
                item = self._label2idx(item)
        self.idx = item
        return super(DArray, self).__getitem__(item)

    def vstack(self, larr):
        return LabeledArray(np.vstack((self, larr)), np.vstack((self.labelarr, larr.labelarr)))

    def hstack(self, larr):
        if (self.labelarr == larr.labelarr).all():
            return LabeledArray(np.hstack((self, larr)), self.labelarr)

    def save(self, file_name):
        np.savez_compressed(file_name, self, self.labelarr)

    def load(self, file_name):
        if not file_name.endswith('.npz'):
            file_name = file_name + '.npz'
        f = np.load(file_name)
        arr, labelarr = f['arr_0'], f['arr_1']
        return LabeledArray(arr, labelarr)


def labeltuple2labelarr(labels):
    labelarr = np.zeros((len(labels), max([len(i) for i in labels])), dtype=object)
    for num, label in enumerate(labels):
        for n1, lab in enumerate(label):
            labelarr[num, n1] = lab
    labelarr[np.where(labelarr == 0)] = ''
    return labelarr

class LabeledArrayConstructor(object):
    def __init__(self, reg):
        self.reg = reg

    def construct(self):
        self._make()
        return self.darray

    def _make(self):
        for obj, ch_dict in self.reg.iteritems():
            for chname, ch_lists in ch_dict.iteritems():
                for rplist in ch_lists:
                    if not hasattr(self, 'darray'):
                        self.darray = self._make_prop(obj, rplist)
                    a = self._make_signal(obj, rplist, str(chname))
                    self.darray = self.darray.vstack(a)

    def _make_prop(self, obj, rplist):
        arr, labels = self._extract_prop(obj, rplist)
        labels, arr = sort_labels_and_arr(labels, arr)
        labelarr = self._convert_labels_to_labelarr(labels)
        return LabeledArray(arr, labelarr)

    def _make_signal(self, obj, rplist, ch):
        arr, labels = self._extract_signal(obj, rplist, ch)
        labelarr = self._convert_labels_to_labelarr(labels)
        return LabeledArray(arr, labelarr)
        
    def _extract_prop(self, obj, rplist):
        arr_list, labels = [], []
        PROPS = ('area', 'centroid', 'eccentricity', 'label', 
                 'major_axis_length', 'minor_axis_length', 'perimeter')
        for pr in PROPS:
            if pr == 'centroid':
                arr_list.append([i[pr][0] for i in rplist])
                arr_list.append([i[pr][1] for i in rplist])
                labels.append([obj, 'centroid0', ''])
                labels.append([obj, 'centroid1', ''])
            else:
                arr_list.append([i[pr] for i in rplist])
                labels.append([obj, pr, ''])
        arr = np.vstack(arr_list)
        return arr, labels

    def _convert_labels_to_labelarr(self, labels):
        labelarr = np.zeros((len(labels), max([len(i) for i in labels])), dtype=object)
        for num, label in enumerate(labels):
            for n1, lab in enumerate(label):
                labelarr[num, n1] = lab
        labelarr[np.where(labelarr==0)] = ''
        return labelarr

    
    def _extract_signal(self, obj, rplist, ch):
        arr_list, labels = [], []
        PROPS = ('mean_intensity', 'median_intensity', 'total_intensity',
                 'std_intensity', 'cv_intensity', 'max_intensity')
        signal = OrderedDict()
        for pr in PROPS:
            signal[pr] = []
        for rp in rplist:
            pix = rp['intensity_image']
            pix = pix[pix != 0]
            signal['mean_intensity'].append(np.mean(pix))
            signal['median_intensity'].append(np.median(pix))
            signal['total_intensity'].append(rp['area'] * np.mean(pix))
            signal['std_intensity'].append(np.std(pix))
            signal['cv_intensity'].append(np.std(pix)/np.mean(pix))
            try:
                signal['max_intensity'].append(np.max(pix))
            except:
                signal['max_intensity'].append(np.nan)
        for pr in PROPS:
            arr_list.append(signal[pr])
            labels.append([obj, ch, pr])
        arr = np.vstack(arr_list)
        return arr, labels
