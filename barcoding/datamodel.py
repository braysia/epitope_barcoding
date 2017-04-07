
from scipy.ndimage import imread
from os.path import join
import numpy as np
from scipy.ndimage import imread
from os.path import join, abspath, curdir, dirname
from glob import glob
from mi_align import MutualInfoAlignerMultiHypothesis as miamh
import json
import tifffile as tiff
from collections import OrderedDict
from skimage.measure import regionprops

class ImagesConstructor(object):
    """assuming corresponding patterns exist in all folders.
    The third axis corresponds to each rounds of staining.
    """
    def __init__(self, folders, patterns, cnames):
        self.folders = folders
        self.patterns = patterns
        self.cnames = cnames

    def construct(self):
        """ return dict
        """
        images = dict()
        for num, (pi, ci) in enumerate(zip(self.patterns, self.cnames)):
            for fn, f in enumerate(self.folders):
                path = self.read_img_path(f, pi)[0]
                img = imread(path)
                if fn == 0:
                    template = np.zeros((img.shape[0], img.shape[1], len(self.folders)))
                template[:, :, fn] = img
            images[ci] = template
        return images

    def read_img_path(self, folder, pattern):
        return glob(join(folder, pattern))


class ImagesCropper(object):
    """receive Images from ImagesConstructor and then crop it.
    Use calc_jitters() first and then crop().
    """

    def __init__(self, images, DOWNSAMPLE=(8, 4, 2)):
        self.images = images
        self.DOWNSAMPLE = DOWNSAMPLE

    def calc_jitters(self):
        """Calculate MI for all channels and pick one alignment with the maximum MI. 
        """
        images = self.images
        jitters = [(0, 0)]  # the first image is the standard
        for dim in range(images[images.keys()[0]].shape[2]-1):
            ji_store, mi_store = [], []
            for ch in images.keys():
                img1 = images[ch][:, :, 0]
                img2 = images[ch][:, :, dim+1]
                mi = miamh(img1, img2, DOWNSAMPLE=self.DOWNSAMPLE)
                mi.execute()
                ji_store.append((mi._j, mi._i))
                mi_store.append(mi.mi)
            print ji_store[mi_store.index(max(mi_store))]
            jitters.append(ji_store[mi_store.index(max(mi_store))])
        self.jitters = jitters

    def crop_multiimages(self, img_list, jit_list):
        """
            img_list (List(np.ndarray)): a list of images
            jit_list (List(tuple)): tuple contains jitters e.g. (-5, 5)

        >> im_list = [np.ones((4, 4)), np.ones((4, 4)), np.ones((4, 4))]
        >> ji_list = [(0, 0), (1, 1), (-1, -1)]
        >> crop_common(im_list, ji_list).shape
        (2, 2, 3)
        """
        IMG_NUM = len(img_list)
        s0, s1 = img_list[0].shape[0], img_list[0].shape[1]
        template = np.ones((s0*3, s1*3, IMG_NUM)) * np.Inf
        for num, (img, jit) in enumerate(zip(img_list, jit_list)):
            template[s0+jit[1]:s0*2+jit[1], s1+jit[0]:s1*2+jit[0], num] = img
        x, y = np.where(-(np.isinf(np.max(template, axis=2))))
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        return template[xmin:xmax+1, ymin:ymax+1, :]

    def crop_images(self, images, jit_list):
        new_images = {}
        for ch, img in images.iteritems():
            imlist = [img[:, :, num] for num in range(img.shape[2])]
            new_images[ch] = self.crop_multiimages(imlist, jit_list)
        return new_images

    def crop(self):
        return self.crop_images(self.images, self.jitters)


class ImagesCropperNuc(ImagesCropper):
    """
    Use DOWNSAMPLE=(8, 4, 2) if not work.
    """
    def __init__(self, images, nucimg, DOWNSAMPLE=(16, 8, 4, 2)):
        self.images = images
        self.nucimg = nucimg
        self.DOWNSAMPLE = DOWNSAMPLE

    def calc_jitters(self):
        """Calculate MI for all channels and pick one alignment with the maximum MI. 
        """
        self._ji_store, self._mi_store = [], []
        images = self.images
        jitters = [(0, 0)]  # the first image is the standard
        for dim in range(images[images.keys()[0]].shape[2]):
            ji_store, mi_store = [], []
            for ch in images.keys():
                img1 = self.nucimg.copy()
                img2 = images[ch][:, :, dim]
                x, y = np.ceil(img1.shape[0]/5), np.ceil(img1.shape[1]/5)
                mi = miamh(img1, img2, DOWNSAMPLE=self.DOWNSAMPLE)
                mi.execute()
                ji_store.append((mi._j, mi._i))
                mi_store.append(mi.mi)
            print ji_store[mi_store.index(max(mi_store))]
            jitters.append(ji_store[mi_store.index(max(mi_store))])
            self._ji_store.append(ji_store)
            self._mi_store.append(mi_store)
        self.jitters = jitters

    def crop_images(self, images, jit_list):
        new_images = ImageDict()
        for num, (ch, img) in enumerate(images.iteritems()):
            imlist = [img[:, :, num] for num in range(img.shape[2])]
            imlist.insert(0, self.nucimg.copy())
            arr = self.crop_multiimages(imlist, jit_list)
            new_images[ch] = arr[:, :, 1:]
            crop_nucimg = arr[:, :, 0]
        return new_images, crop_nucimg


class ImageDict(OrderedDict):
    def __sub__(self, other):
        new = ImageDict()
        for key in self.iterkeys():
            arr = self[key] - other[key]
            arr[arr<0] = 0
            new[key] = arr
        return new

    def stack(self, imdict):
        newdict = self.copy()
        for key in imdict.iterkeys():
            if key not in newdict:
                newdict[key] = imdict[key]
            else:
                if newdict[key].ndim == 3 and imdict[key].ndim == 2:
                    imdict[key] = np.expand_dim(imdict[key], axis=2)
                newdict[key] = np.stack((newdict[key], imdict[key]), axis=2)
        return newdict


    def return_arr_key(self):
        arr_list = []
        key_list = []
        for key, arr in self.iteritems():
            if arr.ndim == 2:
                arr = np.expand_dims(arr, 2)
            arr_list.append(arr)
            for dim in range(arr.shape[2]):
                key_list.append("r{0}_{1}".format(dim, key))
        arr3d = np.rollaxis(np.dstack(arr_list), axis=2)
        return arr3d, key_list

    def save(self, file_name):
        if not file_name.endswith('.tif'):
            file_name = file_name + '.tif'
        arr, keys = self.return_arr_key()
        metadata = json.dumps(dict(shape=arr.shape, keys=keys))
        tiff.imsave(file_name, arr.astype(np.float32), description=metadata)

    def load(self, file_name):
        arr, metadata = self._readfile(file_name)
        keys = metadata["keys"]
        chs = [k.split('_')[-1] for k in keys]
        for num, ch in enumerate(chs):
            if not ch in self:
                self[ch] = arr[:, :, num]
            else:
                self[ch] = np.dstack((self[ch], arr[:, :, num]))

    def _readfile(self, file_name):
        with tiff.TiffFile(file_name) as tif:
            arr = tif.asarray()
            metadata = tif[0].image_description
        arr = np.moveaxis(arr, 0, -1)
        metadata = json.loads(metadata.decode('utf-8'))
        return arr, metadata

class Field(object):
    def __init__(self, folder, patterns, cnames):
        self.folder = folder
        self.patterns = patterns
        self.cnames = cnames

    def construct(self):
        """ return dict
        """
        images = ImageDict()
        for num, (pi, ci) in enumerate(zip(self.patterns, self.cnames)):
            path = self.read_img_path(self.folder, pi)[0]
            images[ci] = imread(path)
        return images

    def read_img_path(self, folder, pattern):
        return glob(join(folder, pattern))


class Unmixer(object):
    def __init__(self, field, weight, weight_channels):
        self.field = field
        self.weight = weight
        self.channels = weight_channels

    def unmix(self):
        stacked = np.vstack([self.field[ch].ravel() for ch in self.channels])
        corrected = np.dot(np.linalg.inv(self.weight), stacked)
        for num, ch in enumerate(self.channels):
            vec = corrected[num, :]
            vec[vec < 0]
            self.field[ch] = vec.reshape(self.field[ch].shape)
        return self.field


def unmixing(field, weight):
    """Does not work if len(field)!=weight.shape[0].
    """
    corr_field = ImageDict()
    stacked = np.vstack([i.ravel() for i in field.itervalues()])
    corrected = np.dot(np.linalg.inv(weight), stacked)
    for num, key in enumerate(field.iterkeys()):
        vec = corrected[num, :]
        vec[vec < 0] = 0
        corr_field[key] = vec.reshape(field[key].shape)
    return corr_field


class RegionExtractor(object):
    def __init__(self, imdict, objnames=['nuc', 'cyto']):
        self.imdict = imdict.copy()
        self.maskdict = OrderedDict()
        for objname in objnames:
            self.maskdict[objname] = self.imdict.pop(objname).astype(np.uint32)

    def _extract(self):
        regions = OrderedDict()
        for obj, mask in self.maskdict.iteritems():
            regions[obj] = OrderedDict()
            for ch, img in self.imdict.iteritems():
                img = np.expand_dims(img, axis=2) if img.ndim ==2 else img
                p = []
                for n1 in range(img.shape[2]):
                    p.append(regionprops(mask, img[:, :, n1]))
                regions[obj][ch] = p
        return regions
