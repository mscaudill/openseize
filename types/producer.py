from collections.abc import Iterable, Generator, Sequence
import abc
import itertools
import numpy as np

from openseize.types import mixins

def producer(data, chunksize, axis=-1):
    """Returns an iterable of numpy ndarrays from an ndarray, a sequence of
    arrays or generator of arrays.

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
        
    Returns: a Producer iterable   
    """

    if isinstance(data, Producer):
        data.chunksize = chunksize
        data.axis = axis
        return data
    if isinstance(data, Generator):
        return _ProducerFromGenerator(data, chunksize, axis)
    elif isinstance(data, np.ndarray):
        return _ProducerFromArray(data, chunksize, axis)
    elif isinstance(data, Sequence):
        data = np.array(data)
        return _ProducerFromArray(data, chunksize, axis)
    else:
        msg = 'unproducible type: {}'
        raise TypeError(msg.format(type(data)))


class Producer(Iterable, mixins.ViewInstance):
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
        for start, stop in self.segments():
            #find generator segments containing start through stop (+1)
            genstart, genstop = np.searchsorted(datapts, [start, stop])
            genstop += 1
            #create 2 indpt generators from data so one can be sliced
            self.data, tmp = itertools.tee(self.data, 2)
            sliced = itertools.islice(tmp, genstart, genstop)
            arrs = [arr for arr in sliced]
            #data exhaustion check
            if not arrs:
                break
            arr = np.concatenate(arrs, axis=self.axis)
            #index within start data segement where producer start occurs
            idx = max(start - datapts[genstart - 1], 0)
            #slice collected data segments (arr) starting from index & yield
            slices = [slice(None)] * arr.ndim
            slices[self.axis] = slice(idx, idx + self.chunksize)
            yield arr[tuple(slices)]



if __name__ == '__main__':
   

    def g(chs=4, samples=100021, csize=100, seed=0):
        """ """

        np.random.seed(seed)
        starts = range(0, samples, csize)
        segments = itertools.zip_longest(starts, starts[1:],
                                         fillvalue=samples)
        for start, stop in segments:
            arr = np.random.random((chs, stop-start))
            yield arr

    values = np.concatenate([arr for arr in g(seed=0)], axis=-1)

    a = producer(g(seed=0), chunksize=25, axis=-1)
    y = np.concatenate([arr for arr in a], axis=-1)
    print(np.allclose(values, y))

    """
    x = np.random.random((10, 4,1100))
    gen = (arr for arr in np.split(x, 10, axis=-1))
    gprod = producer(gen, chunksize=30, axis=-1)
    y = np.concatenate([arr for arr in gprod], axis=-1)
    print(np.allclose(x,y))
    """



