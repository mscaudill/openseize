from collections.abc import Iterable
import itertools
import numpy as np

from openseize.mixins import ViewInstance

class Producer(Iterable):
    """ """

    def __init__(self, data, chunksize):
        """ """

        self.data = data
        self.chunksize = chunksize


class ArrayProducer(Producer):
    """ """

    def __init__(self, data, chunksize, axis=-1):
        """ """

        super().__init__(data, chunksize)
        self.axis = axis

    @property
    def segments(self):
        """ """

        extent = self.data.shape[self.axis]
        starts = range(0, extent, self.chunksize)
        return itertools.zip_longest(starts, starts[1:], fillvalue=extent)

    def __iter__(self):
        """ """

        slices = [slice(None)] * self.data.ndim
        for segment in self.segments:
            slices[self.axis] = slice(*segment)
            yield self.data[tuple(slices)]


class GenProducer(Producer):
    """ """

    def __init__(self, data, chunksize, axis=-1):
        """ """

        super().__init__(data, chunksize)
        self.axis = axis
        
    @property
    def genchunksize(self):
        """ """
        
        #make 2 indpt generators
        self.data, tmp = itertools.tee(self.data, 2)
        return next(tmp).shape[self.axis]

    def __iter__(self):
        """ """
       
        gsize = self.genchunksize
        segments = zip(itertools.count(0, self.chunksize), 
                       itertools.count(self.chunksize, self.chunksize))
        for idx, (start, stop) in enumerate(segments):
            #compute generator segments containing segment pts
            gsegments = (start // gsize, stop // gsize + 1)
            self.data, tmp = itertools.tee(self.data, 2)
            g = itertools.islice(tmp, *gsegments)
            arrs = [arr for arr in g]
            if not arrs:
                break
            arr = np.concatenate(arrs, axis=self.axis)
            slices = [slice(None)] * arr.ndim
            leftbound = start - (start // gsize * gsize)
            slices[self.axis] = slice(leftbound, leftbound + self.chunksize)
            yield arr[tuple(slices)]




if __name__ == '__main__':
   

    """
    x = np.random.random((4,100))
    a = ArrayProducer(x, chunksize=23, axis=-1)
    y = np.concatenate([arr for arr in a], axis=1)
    print(np.allclose(x,y))
    """

    x = np.random.random((4,1100))
    gen = (arr for arr in np.split(x, 11, axis=-1))
    gprod = GenProducer(gen, 32)
    y = np.concatenate([arr for arr in gprod], axis=1)
    print(np.allclose(x,y))



