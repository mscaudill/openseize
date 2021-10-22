from collections.abc import Reversible, Sequence
import abc
import copy
import itertools
import numpy as np

from openseize.types import mixins
from openseize.tools import arraytools

def producer(obj, chunksize, axis, mask=None, **kwargs):
    """Returns an reversible iterable of numpy ndarrays from an ndarray, 
    a sequence of arrays or generator of arrays.

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
        mask (1-D bool):                       boolean array to mask
                                               producer outputs along axis.
                                               Values of obj at True mask
                                               indices will be yielded and
                                               False will be ignored. If 
                                               mask len does not match obj
                                               len along axis, The producer
                                               yields subarrays upto the 
                                               shorter of mask and obj.
        
    Note: if obj is generator type it must be a callable (i.e. generator
    function) that returns a generator object.
    
    Returns: a Producer reversible iterable   
    """

    if isinstance(obj, Producer):
        obj.chunksize = int(chunksize)
        obj.axis = axis
        result = obj

    elif callable(obj):
        result = _ProduceFromGenerator(obj, chunksize, axis, **kwargs)

    elif isinstance(obj, np.ndarray):
        result = _ProduceFromArray(obj, chunksize, axis, **kwargs)

    elif isinstance(obj, Sequence):
        data = np.array(obj)
        result = _ProduceFromArray(obj, chunksize, axis, **kwargs)

    else:
        msg = 'unproducible type: {}'
        raise TypeError(msg.format(type(obj)))

    #apply mask if passed
    if mask is None:
        return result
    else:
        return _MaskedProducer(result, mask, chunksize, axis, **kwargs)


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
        self._chunksize = int(chunksize)
        self.axis = axis
        self.__dict__.update(kwargs)
        #ViewInstance will use Collector name not subclass name
        self.__class__.__name__ = 'Producer'

    @property
    def chunksize(self):
        """Returns the chunksize of this Producer."""

        return self._chunksize

    @chunksize.setter
    def chunksize(self, value):
        """Sets this Producer's chunksize attr ensuring it is int type."""

        self._chunksize = int(value)


class _ProduceFromArray(Producer):
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

    @property
    def genlen(self):
        """Returns the number of subarrays in the generator."""

        if hasattr(self, '_genlen'):
            #get stored value
            return self._genlen
        else:
            #compute num of arrays in generator obj
            self._genlen = sum(1 for _ in self.data())
            return self._genlen

    def __partitioner(self, gen):
        """An iterator of arrays of chunksize along this Producers axis
        obtained from a generator yielding arrays of any size along axis.

        Args:
            gen (generator):        a generator object that yields ndarrays
        """

        #collect arrays and overage amt until chunksize reached
        collected, size = list(), 0
        for subarr in gen:
            collected.append(subarr)
            #check if chunksize has been reached
            size += subarr.shape[self.axis]
            if size >= self.chunksize:
                # split the collected storing overage for next round
                y = np.concatenate(collected, axis=self.axis)
                y, overage = np.split(y, [self.chunksize], axis=self.axis)
                yield y
                # exhaust overage while its size > chunksize
                while overage.shape[self.axis] >= self.chunksize:
                    y, overage = np.split(overage, [self.chunksize],
                                          axis=self.axis)
                    yield y
                #reset collected and size, and carry overage to next round
                collected = []
                collected.append(overage)
                size = overage.shape[self.axis]
        else:
            #yield everything that is left
            yield np.concatenate(collected, axis=self.axis)

    def __iter__(self):
        """Returns an iterator yielding ndarrays of chunksize along axis."""

        #build gen obj
        gen = self.data()
        return self.__partitioner(gen)

    def reverse_generator(self):
        """A generating function that yields elements from this Producer's
        generating function in reverse order. """

        #closure to track generator position starting at len of gen
        idx = self.genlen
        def reverser():
            nonlocal idx
            while idx > 0:
                arr = next(itertools.islice(self.data(), idx-1, idx))
                idx -= 1
                yield np.flip(arr, self.axis)
        return reverser()

    def __reversed__(self):
        """Returns an iterator of ndarrays of chunksize along axis in 
        reverse order."""

        #build a reverse generator obj
        rgen = self.reverse_generator()
        return self.__partitioner(rgen)


class _MaskedProducer(Producer):
    """A Producer of numpy arrays with values that have been filtered by
    a boolean mask.

    Args:
       pro (obj):               reversible iterable producing numpy ndarrays
       mask (1-D Bool):         a 1-D boolean array of masked values. The
                                length of the mask does not have to match 
                                the length of the producer but 
                                MaskedProducer will stop producing as soon
                                as the producer or mask raise StopIteration.
        chunksize (int):        size of ndarray along axis to return per
                                iteration
        axis (int):             producer's axis along which data is chunked

    Note: The bool mask is applied on all 1-D slices of producer's array
    oriented along axis (see np.take).
    """

    def __init__(self, pro, mask, chunksize, axis, **kwargs):
        """Initialize this Producer with a boolean array mask."""

        super().__init__(pro, chunksize, axis, **kwargs)
        self.mask = producer(mask, chunksize, axis=0, *kwargs)

    @property
    def chunksize(self):
        """Returns the chunksize of this MaskedProducer."""

        return self.data._chunksize

    @chunksize.setter
    def chunksize(self, value):
        """On change, set chunksize for both producer and mask."""

        self.data._chunksize = int(value)
        self.mask._chunksize = int(value)

    def __iter__(self):
        """Returns an iterator of boolean masked numpy arrays along axis."""

        for (arr, marr) in zip(self.data, self.mask):
            indices = np.flatnonzero(marr)
            yield np.take(arr, indices, axis=self.axis)

    def __reversed__(self):
        """Returns a reversed iterator of bool masked arrays along axis."""

        for arr, marr in zip(reversed(self.data), reversed(self.mask)):
            indices = np.flatnonzero(marr)
            yield np.take(arr, indices, axis=self.axis)

