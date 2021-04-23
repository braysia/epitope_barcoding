from collections import OrderedDict


class ImageDict(OrderedDict):
    def __sub__(self, other):
        new = ImageDict()
        for key in self.iterkeys():
            arr = self[key] - other[key]
            # arr[arr<0] = 0
            new[key] = arr
        return new