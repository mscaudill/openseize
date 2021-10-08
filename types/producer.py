from collections.abc import Reversible, Generator, Sequence
import abc
import copy
import itertools
import numpy as np

from openseize.types import mixins

def producer(obj, chunksize, axis=-1):
    """Returns an iterable of numpy ndarrays from an ndarray, a sequence of
    arrays or generator of arrays.

    Args:
        obj (ndarray, Sequence, Generator):   an ndarray or object that
                                               contains or yields ndarrays
                                               (e.g. np.memmap, list of
                                               ndarrays, or a generator of
                                               ndarrays of any shape)
        chunksize (int):                       len of each ndarray yielded
                                               by this iterable along axis
                                               parameter
        axis (int):                            axis of data to partition
                                               into chunks of chunksize
        
    Returns: a Producer iterable   
    """

    if isinstance(obj, Producer):
        #copy underlying data
        obj.chunksize = chunksize
        obj.axis = axis
        return obj

        """
        # This does not work because advancing the subgen advances the
        # original gen SOLVE THIS OR THROW ERROR
        if isinstance(obj.data, Generator, itertools._tee):
            data, old = itertools.tee(self.data, 2)
            obj.data = data
        elif isinstance(obj.data, np.ndarray):
            obj.data = np.copy(pro.data)
        elif isinstance(obj.data, Sequence):
            obj.data = copy.deepcopy(obj.data)
        return obj
        """

    if isinstance(obj, (Generator, itertools._tee)):
        return _ProducerFromGenerator(obj, chunksize, axis)

    elif isinstance(obj, np.ndarray):
        return _ProducerFromArray(obj, chunksize, axis)

    elif isinstance(obj, Sequence):
        data = np.array(data)
        return _ProducerFromArray(obj, chunksize, axis)

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
    instance use producer() function call.
    """

    def __init__(self, data, chunksize, axis, **kwargs):
        """Concrete initializer for all Producer subclasses."""

        self.data = data
        self.chunksize = int(chunksize)
        self.axis = axis
        self.__dict__.update(kwargs)
        #ViewInstance will use Producer name not subclass name
        self.__class__.__name__ = 'Producer'

    @abc.abstractproperty
    def shape(self):
        """Returns the shape of this Producer data attr."""


class _ProducerFromArray(Producer):
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


class _ProducerFromGenerator(Producer):
    """A Producer of ndarrays from a generator of ndarrays.

    A generator of ndarrays will return arrays whose shape is determined
    when the generator was created. This iterable moves over these arrays
    collecting arrays until chunksize shape is reached along axis. Once
    reached, iter yields a collected ndarray.
    """

    @property
    def shape(self):
        """Returns the summed shape across all ndarrays yielded by data."""

        #create 2 indpt gens & overwrite data since tmp will advance it
        self.data, tmp = itertools.tee(self.data, 2)
        #advance by one to get intial shape then update
        result = np.array(next(tmp).shape)
        for arr in tmp:
            result[self.axis] += arr.shape[self.axis]
        return result
 
    def _datapts(self):
        """Returns start indices of each yielded array from this Producer's
        generator.

        The generator may yield arrays of unequal len along the chunking 
        axis. So track start indices of each yielded ndarray.
        """

        #create 2 indpt gens & overwrite data since tmp will advance it
        self.data, tmp = itertools.tee(self.data, 2)
        return np.cumsum([x.shape[self.axis] for x in tmp])

    def segments(self):
        """Returns an iterable of start, stop tuples of span chunksize."""

        #since data gen len is unknowable create infinite segments
        return zip(itertools.count(0, self.chunksize), 
                   itertools.count(self.chunksize, self.chunksize))

    def __iter__(self):
        """Returns an iterator yielding ndarrays of chunksize along axis."""

        datapts = self._datapts()
        print(datapts)
        for start, stop in self.segments():
            print(start, stop)
            #find generator segments containing start through stop (+1)
            genstart, genstop = np.searchsorted(datapts, [start, stop],
                    side='right')
            #create 2 indpt generators from data so one can be sliced
            self.data, tmp = itertools.tee(self.data, 2)
           
            genstop = max(genstop, genstart + 1)
            print('gen start, stop = ', genstart, genstop)
            sliced = itertools.islice(tmp, genstart, genstop)
            arrs = [arr for arr in sliced]
            print([a.shape for a in arrs])
            #data exhaustion check
            if not arrs:
                return
            arr = np.concatenate(arrs, axis=self.axis)
            #offset within start gen segement where producer start occurs

            #FIXME THIS IS A MESS!
            #idx = start - datapts[genstart - 1] if genstart > 0 else start
            idx = start if start < datapts[genstart-1] else start - datapts[genstart
                    ]-start
            print('offset = ', idx)
            #slice collected data segments (arr) starting from index & yield
            slices = [slice(None)] * arr.ndim
            slices[self.axis] = slice(idx, idx + self.chunksize)
            yield arr[tuple(slices)]

    def __reversed__(self):
        """Returns an iterator yielding ndarrays of chunnksize along axis in
        reverse order."""

        datapts = self._datapts()
        #print('datapts = ', datapts)
        csize = self.chunksize
        samples = self.shape[self.axis]
        #get reverse segments
        rsegments = zip(itertools.count(samples, -1 * csize),
                        itertools.count(samples - csize, -1 * csize))
        
        for start, stop in rsegments:
            print('producer start, stop = ', start, stop)
            #exhaustion check
            if start <= 0:
                print('reverse break')
                return
            #find the generator segments containing stop trhough start
            genstart, genstop = np.searchsorted(datapts, [start, stop])
            print('gen start, stop = ', genstart, genstop)
            #create 2 indpt generators from data so one can be sliced
            self.data, tmp = itertools.tee(self.data, 2)
            sliced = itertools.islice(tmp, genstop, genstart + 1)
            arrs = [arr for arr in sliced]
            arr = np.concatenate(arrs, axis=self.axis)
            #BETA
            arr = np.flip(arr, axis=self.axis)
            #offset
            idx = datapts[genstart] - start
            #print('offset = ', idx)

            slices = [slice(None)] * arr.ndim
            slices[self.axis] = slice(idx, idx + self.chunksize)
            #print('slices = ', slices)
            y = arr[tuple(slices)]
            #print('y shape = ', y.shape)
            yield y

            

if __name__ == '__main__':
   
    """
    def g(chs=4, samples=50000, csize=10000, seed=0):
        """ """

        np.random.seed(seed)
        starts = range(0, samples, csize)
        segments = itertools.zip_longest(starts, starts[1:],
                                         fillvalue=samples)
        for start, stop in segments:
            arr = np.random.random((chs, stop-start))
            yield arr

    values = np.concatenate([arr for arr in g(seed=0)], axis=-1)

    a = producer(g(seed=0), chunksize=10000, axis=-1)
    y = np.concatenate([arr for arr in a], axis=-1)
    print(np.allclose(values, y))
    """

    """
    b = reversed(a)
    z = np.concatenate([arr for arr in b], axis=-1)
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

    def test_gen():
        """Test if producer produces correct subarrays from a generator."""

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

