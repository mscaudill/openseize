from collections.abc import Reversible, Sequence
import abc
import copy
import itertools
import numpy as np

from openseize.types import mixins
from openseize.tools import arraytools

def producer(obj, chunksize, axis, shape=None):
    """Returns an reversible iterable of numpy ndarrays from an ndarray, 
    a sequence of arrays or generator of arrays.

    Note: if obj is generator type it must be a callable (i.e. generator
    function) that returns a generator object.

    Args:
        obj (ndarray, Sequence, Producer):     an ndarray or object that
                                               contains or yields ndarrays
                                               (e.g. np.memmap, list of
                                               ndarrays, or a generator of
                                               ndarrays of any shape)
        chunksize (int):                       len of each ndarray yielded
                                               by this iterable along axis
                                               parameter
        axis (int):                            axis of data to partition
                                               into chunks of chunksize
        shape (tuple):                         shape of 
        
    Returns: a Producer reversible iterable   
    """

    if isinstance(obj, Producer):
        obj.chunksize = int(chunksize)
        obj.axis = axis
        return obj

    if callable(obj):
        return _ProduceFromGenerator(obj, chunksize, axis, shape)

    elif isinstance(obj, np.ndarray):
        return _ProduceFromArray(obj, chunksize, axis)

    elif isinstance(obj, Sequence):
        data = np.array(data)
        return _ProduceFromArray(obj, chunksize, axis)

    else:
        msg = 'unproducible type: {}'
        raise TypeError(msg.format(type(obj)))


class Producer(Reversible, mixins.ViewInstance):
    """ABC defining concrete and required methods of all Producers.

    Args:
        data (ndarray, Sequence, Generator):   an ndarray or object that
                                               contains or yields ndarrays
                                               (e.g. np.memmap, list of
                                               ndarrays, or a generator of
                                               ndarrays of any shape)
        chunksize (int):                       len of each ndarray yielded
                                               by this iterable along axis
                                               parameter
        axis (int):                            axis of data to partition
                                               into chunks of chunksize

    As an ABC, this class cannot be instantiated. To create a producer
    instance use producer function.
    """

    def __init__(self, data, chunksize, axis, **kwargs):
        """Concrete initializer for all Collector subclasses."""

        self.data = data
        self.chunksize = int(chunksize)
        self.axis = axis
        self.__dict__.update(kwargs)
        #ViewInstance will use Collector name not subclass name
        self.__class__.__name__ = 'Producer'

    @abc.abstractproperty
    def shape(self):
        """Returns the shape of this Producer's data attr."""


class _ProduceFromArray(Producer):
    """A Producer of ndarrays from an ndarray."""

    @property
    def shape(self):
        """Returns the shape of this Producer's data array attr."""

        return self.data.shape

    def segments(self):
        """Returns start, stop indices of span chunksize to apply along
        this Producer's axis."""

        extent = self.data.shape[self.axis]
        starts = range(0, extent, self.chunksize)
        return itertools.zip_longest(starts, starts[1:], fillvalue=extent)

    def __iter__(self):
        """Returns an iterator yielding ndarrays of chunksize along axis."""

        #create an ndarray slice sequence
        slices = [slice(None)] * self.data.ndim
        for segment in self.segments():
            #update slice and fetch from data
            slices[self.axis] = slice(*segment)
            yield self.data[tuple(slices)]

    def __reversed__(self):
        """Returns an iterator yielding ndarrays of chunksize starting from
        the end of data along axis."""

        #create an ndarray slice sequence
        slices = [slice(None)] * self.data.ndim
        for start, stop in reversed(list(self.segments())):
            #update slice, fetch and flip
            slices[self.axis] = slice(start, stop)
            y = np.flip(self.data[tuple(slices)], axis=self.axis)
            yield y


class _ProduceFromGenerator(Producer):
    """A Producer of ndarrays from a generating function of ndarrays.

    This iterable moves over a generator's yielded arrays collecting until
    chunksize shape is reached along axis. Once reached, it yields a 
    collected ndarray.
    """

    def __init__(self, genfunc, chunksize, axis, shape):
        """ """

        self.genfunc = genfunc
        self.chunksize = int(chunksize)
        self.axis = axis
        self._shape = shape
        #ViewInstance will use Collector name not subclass name
        self.__class__.__name__ = 'Producer'

    @property
    def shape(self):
        """Returns the summed shape across all ndarrays yielded by pro."""

        return self._shape

    def __iter__(self):
        """Returns an iterator yielding ndarrays of chunksize along axis."""

        #build gen obj
        gen = self.genfunc()
        #collect arrays and overage amt until chunksize reached
        collected, size = [np.zeros((self.shape[0], 0))], 0
        for subarr in gen:
            collected.append(subarr)
            #check if chunksize has been reached
            size += subarr.shape[self.axis]
            if size >= self.chunksize:
                # split the collected storing overage for next round
                y = np.concatenate(collected, axis=self.axis)
                print('unsliced shape = ', y.shape)
                y, overage = np.split(y, [self.chunksize], axis=self.axis)
                #reset collected and size
                collected = []
                collected.append(overage)
                size = overage.shape[self.axis]
                yield y
        else:
            #yield everything that is left
            yield np.concatenate(collected, axis=self.axis)

    def __reversed__(self):
        """Returns an iterator yielding ndarrays of chunnksize along axis in
        reverse order."""

        #definitely going to need islice to do this
        


    
    def _datapts(self):
        """Returns start indices of each yielded array from this Producer's
        generator.

        The generator may yield arrays of unequal len along the chunking 
        axis. So track start indices of each yielded ndarray.
        """

        return np.cumsum([x.shape[self.axis] for x in self.data])

    def segments(self):
        """Returns an iterable of start, stop tuples of span chunksize."""

        #since data gen len is unknowable create infinite segments
        return zip(itertools.count(0, self.chunksize), 
                   itertools.count(self.chunksize, self.chunksize))

    def DEPR__iter__(self):
        """Returns an iterator yielding ndarrays of chunksize along axis."""


        #FIXME REMOVE REFS TO TEE
        datapts = self._datapts()
        for start, stop in self.segments():
            #exhaustion check
            if start >= datapts[-1]:
                return
            #find generator segments containing start through stop
            a, b = np.searchsorted(datapts, [start, stop], side='right')
            b += 1
            #create 2 indpt generators from data so one can be sliced
            self.data, tmp = itertools.tee(self.data, 2)
            sliced = itertools.islice(tmp, a, b)
            arrs = [arr for arr in sliced]
            arr = np.concatenate(arrs, axis=self.axis)
            #insert 0 into datapts to get start index of each segment
            seg_starts = np.insert(datapts,0,0)
            # measure offset of producer start to segment start
            offset = start - seg_starts[a]
            #slice from offset and yield
            slices = [slice(None)] * arr.ndim
            slices[self.axis] = slice(offset, offset + self.chunksize)
            yield arr[tuple(slices)]

    def __reversed__(self):
        """Returns an iterator yielding ndarrays of chunnksize along axis in
        reverse order."""

        #FIXME REMOVE REFS TO TEE
        datapts = self._datapts()
        csize = self.chunksize
        samples = self.shape[self.axis]
        #create reverse segments
        rsegments = zip(itertools.count(samples, -1 * csize),
                        itertools.count(samples - csize, -1 * csize))
        for start, stop in rsegments:
            #exhaustion check
            if start <= 0:
                return
            #find the generator segments containing stop trhough start
            genstart, genstop = np.searchsorted(datapts, [start, stop])
            genstart += 1
            #create 2 indpt generators from data so one can be sliced
            self.data, tmp = itertools.tee(self.data, 2)
            sliced = itertools.islice(tmp, genstop, genstart)
            arrs = [arr for arr in sliced]
            arr = np.concatenate(arrs, axis=self.axis)
            #flip array since we read stop to start where stop < start
            arr = np.flip(arr, axis=self.axis)
            #insert 0 into datapts to get start index of each segment
            seg_starts = np.insert(datapts,0,0)
            #compute offset
            idx = seg_starts[genstart] - start
            #slice from offset and yield
            slices = [slice(None)] * arr.ndim
            slices[self.axis] = slice(idx, idx + self.chunksize)
            y = arr[tuple(slices)]
            yield y

            

if __name__ == '__main__':

    from functools import partial
 
    def g(chs=4, samples=50000, segsize=2009, seed=0):
        """ """

        np.random.seed(seed)
        starts = range(0, samples, segsize)
        segments = itertools.zip_longest(starts, starts[1:],
                                         fillvalue=samples)
        for start, stop in segments:
            arr = np.random.random((chs, stop-start))
            yield arr

    values = np.concatenate([arr for arr in g(seed=0)], axis=-1)

    gen = partial(g)
    coll = _CollectFromGenerator(gen, chunksize=10000, axis=-1,
            shape=(4,50000))
    result = np.concatenate([a for a in coll], axis=-1)
    print(np.allclose(result, values))
    

    """
    b = reversed(a)
    z = np.concatenate([arr for arr in b], axis=-1)
    print(np.allclose(values[:,::-1], z))
    """
    

    """
    x = np.random.random((10, 4,1100))
    gen = (arr for arr in np.split(x, 10, axis=-1))
    gprod = producer(gen, chunksize=30, axis=-1)
    y = np.concatenate([arr for arr in gprod], axis=-1)
    print(np.allclose(x,y))
    """

    """
    data = np.random.random((2, 50000))
    pro = producer(data, chunksize=1000, axis=-1)
    rev_gen = reversed(pro)
    rev = np.concatenate([arr for arr in rev_gen], axis=-1)
    print(np.allclose(data[:,::-1], rev))
    """
    
    """
    arrs = [np.random.random((2,10000)) for _ in range(5)]
    gen = (arr for arr in arrs)
    pro = producer(gen, chunksize=1000, axis=-1)
    rev_gen = reversed(pro)
    rev_arrs = [arr for arr in rev_gen]
    rev = np.concatenate(rev_arrs, axis=-1)
    data = np.concatenate([arr for arr in arrs], axis=-1)
    print(np.allclose(data[:,::-1], rev))
    """

    """
    def test_gen():
        #Test if producer produces correct subarrays from a generator.

        chunksize=12231
        np.random.seed(9634)
        #make sure to use subarrays of varying lens along chunking axis
        lens = np.random.randint(2000, high=40000, size=50)
        #keep the arrays for comparison and make a 1-time generator
        arrs = [np.random.random((17, l)) for l in lens]
        gen = (arr for arr in arrs)
        #create a producer from the generator
        pro = producer(gen, chunksize=chunksize, axis=-1)
        #fetch and store the arrays from the producer
        pro_arrays = [arr for arr in pro]
        #create arrays for comparison from original and from producer
        arr = np.concatenate(arrs, axis=-1)
        pro_arr = np.concatenate(pro_arrays, axis=-1)
        print(np.allclose(arr, pro_arr))


    test_gen()
    """

