from collections.abc import Iterable, Generator, Sequence
import itertools
import numpy as np

from openseize.mixins import ViewInstance

def producer(data, chunksize, axis=-1, **kwargs):
    """ """

    if isinstance(data, Generator):
        return _FromGenerator(data, chunksize, axis, **kwargs)
    elif isinstance(data, np.ndarray):
        return _FromArray(data, chunksize, axis, **kwargs)
    elif isinstance(data, Sequence):
        data = np.array(data)
        return _FromArray(data, chunksize, axis, **kwargs)
    else:
        msg = 'unproducible type: {}'
        raise TypeError(msg.format(type(data)))

class _FromArray(ViewInstance):
    """ """

    def __init__(self, data, chunksize, axis, **kwargs):
        """ """

        self.data = data
        self.chunksize = chunksize
        self.axis = axis
        self.__dict__.update(kwargs)
        self.__class__.__name__ = 'Producer'

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


class _FromGenerator(ViewInstance):
    """ """
 
    def __init__(self, data, chunksize, axis, **kwargs):
        """ """

        self.data = data
        self.chunksize = chunksize
        self.axis = axis
        self.__dict__.update(kwargs)
        self.__class__.__name__ = 'Producer'
   
    @property
    def _datachunksize(self):
        """ """
        
        #make 2 indpt generators
        self.data, tmp = itertools.tee(self.data, 2)
        return next(tmp).shape[self.axis]

    def __iter__(self):
        """ """
       
        gsize = self._datachunksize
        segments = zip(itertools.count(0, self.chunksize), 
                       itertools.count(self.chunksize, self.chunksize))
        for start, stop in segments:
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
    ls = x.tolist()
    a = producer(ls, chunksize=23, axis=-1)
    y = np.concatenate([arr for arr in a], axis=1)
    print(np.allclose(x,y))
    """


    x = np.random.random((4,1110))
    gen = (arr for arr in np.split(x, 10, axis=-1))
    gprod = producer(gen, chunksize=30, axis=-1)
    y = np.concatenate([arr for arr in gprod], axis=1)
    print(np.allclose(x,y))



