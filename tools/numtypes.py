from collections.abc import Iterable, Generator, Sequence
import itertools
import numpy as np

from openseize.mixins import ViewInstance

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

    chunksize = int(chunksize)
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


class Producer(Iterable, ViewInstance):
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
        self.chunksize = chunksize
        self.axis = axis
        self.__dict__.update(kwargs)
        #ViewInstance will use Producer name not subclass name
        self.__class__.__name__ = 'Producer'


class _ProducerFromArray(Producer):
    """A Producer of ndarrays from an ndarray."""

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
 
    def _datastep(self):
        """Returns the length of data generator's ndarrays along axis."""
        
        #create 2 indpt gens & overwrite data since tmp will advance it
        self.data, tmp = itertools.tee(self.data, 2)
        #advance an indpt tmp generator by one step to get shape
        return next(tmp).shape[self.axis]

    def segments(self):
        """Returns an iterable of start, stop tuples of span chunksize."""

        #since data gen len is unknowable create infinite segments
        return zip(itertools.count(0, self.chunksize), 
                   itertools.count(self.chunksize, self.chunksize))

    def __iter__(self):
        """Returns an iterator yielding ndarrays of chunksize along axis."""
       
        datastep = self._datastep()
        for start, stop in self.segments():
            #find the data segments containing this Producer segment
            datasegments = (start // datastep, stop // datastep + 1)
            #create 2 indpt generators from data so one can be sliced
            self.data, tmp = itertools.tee(self.data, 2)
            sliced = itertools.islice(tmp, *datasegments)
            arrs = [arr for arr in sliced]
            #data exhaustion check
            if not arrs:
                break
            arr = np.concatenate(arrs, axis=self.axis)
            #index within start data segement where producer start occurs
            idx = start - (start // datastep * datastep)
            #slice collected data segments (arr) starting from index & yield
            slices = [slice(None)] * arr.ndim
            slices[self.axis] = slice(idx, idx + self.chunksize)
            yield arr[tuple(slices)]



if __name__ == '__main__':
   

    """ 
    x = np.random.random((4,100))
    ls = x.tolist()
    a = producer(ls, chunksize=23, axis=-1)
    y = np.concatenate([arr for arr in a], axis=1)
    print(np.allclose(x,y))
    """


    x = np.random.random((10, 4,1110))
    gen = (arr for arr in np.split(x, 10, axis=-1))
    gprod = producer(gen, chunksize=30, axis=-1)
    y = np.concatenate([arr for arr in gprod], axis=-1)
    print(np.allclose(x,y))



