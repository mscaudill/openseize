import numpy as np
import abc
import functools
import inspect
from collections.abc import Iterable, Sequence, Generator
from itertools import zip_longest

from openseize.types import mixins

def producer(obj, chunksize, axis, mask=None, **kwargs):
    """Returns an iterable of numpy ndarrays from an ndarray, a sequence of
    ndarrays or a generating function yielding ndarrays.

    Args:
        obj (ndarray, Sequence, 
             Producer, or generating 
             function of ndarrays):            an ndarray or object that
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
        
    Returns: a Producer iterable   
    """

    if isinstance(obj, Producer):
        obj.chunksize = int(chunksize)
        obj.axis = axis
        result = obj

    elif inspect.isgeneratorfunction(obj):
        shape = kwargs.pop('shape', None)
        if not shape:
            msg = ("producing from a generating func. requires a 'shape' "
                   "keyword argument.")
            raise TypeError(msg)
        result = _ProduceFromGenerator(obj, chunksize, axis, shape, 
                                       **kwargs)

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

def as_producer(func):
    """Decorator that returns a Producer from a generating function.

    Producing from a generator requires the original generating function
    since iteration by a Producer exhaust single use generators. This
    decorator can be used to decorate any generating function recasting it
    as a Producer.

    Implementation note: the first argument of the func to be decorated is
    required to be of type Producer.
    """

    if not inspect.isgeneratorfunction(func):
        msg = 'as_producer requries a generating function not {}'
        raise TypeError(msg.format(type(func)))

    
    @functools.wraps(func)
    def decorated(pro, *args, **kwargs):
        """Returns a producer using values from generating func."""
   
        if not isinstance(pro, Producer):
            msg = ("First positional argument of decorated function"
                   " must be of type {} not {}")
            raise TypeError(msg.format('Producer', type(pro)))

        genfunc = functools.partial(func, pro, *args, **kwargs)
        return producer(genfunc, pro.chunksize, pro.axis, shape=pro.shape,
                        **kwargs)

    return decorated


class Producer(Iterable, mixins.ViewInstance):
    """ABC defining concrete and required methods of all Producers.

    Attrs:
        data (ndarray, Sequence, 
              Producer, or generating
              function yielding ndarrays):     an ndarray or object that
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

    @abc.abstractproperty
    def shape(self):
        """Returns the shape of this producers data attribute."""



class _ProduceFromArray(Producer):
    """A Producer of ndarrays from an ndarray."""

    @property
    def shape(self):
        """Returns the shape of this Producers data array."""

        return self.data.shape

    def _slice(self, start, stop, step=None):
        """Returns a slice tuple for slicing data between start & stop with
        an optional step."""

        slices = [slice(None)] * self.data.ndim
        slices[self.axis] = slice(start, stop, step)
        return tuple(slices)

    def __iter__(self):
        """Returns an iterator yielding ndarrays of chunksize along axis."""

        starts = range(0, self.data.shape[self.axis], self.chunksize)
        for segment in zip_longest(starts, starts[1:], fillvalue=None):
            yield self.data[self._slice(*segment)]


class _ProduceFromGenerator(Producer):
    """A Producer of ndarrays from a generating function of ndarrays.

    This iterable moves over a generator's yielded arrays collecting until
    chunksize shape is reached along axis. Once reached, it yields a 
    collected ndarray.
    """

    def __init__(self, data, chunksize, axis, shape, **kwargs):
        """Initialize this producer with an additional required shape."""

        super().__init__(data, chunksize, axis, **kwargs)
        self._shape = shape

    @property 
    def shape(self):
        """Returns the summed shape of arrays in this Producer."""

        return self._shape

    def __iter__(self):
        """Returns an iterator yielding ndarrays of chunksize along axis."""

        #collect arrays and overage amt until chunksize reached
        collected, size = list(), 0
        for subarr in self.data():
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


class _MaskedProducer(Producer):
    """A Producer of numpy arrays with values that have been filtered by
    a boolean mask.

    Args:
       pro (obj):               iterable producing numpy ndarrays
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
