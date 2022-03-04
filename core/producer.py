"""Tools for creating iterables of ndarrays from numpy arrays, sequences,
binary file readers, and generating functions.

This module contains the following classes and functions:

    producer:
        A function that transforms data from an ndarray, sequence, file
        reader or generating function of ndarrays into an iterable of
        ndarrays.

        Typical usage example:
        pro = producer(data, chunksize=1000, axis=-1)
        #returns an iterable yielding ndarrays of size 1000 along last axis
        from a data source. Data may be an array, a sequence, a file reader
        or a generating function yielding ndarrays.

    as_producer:
        A higher-order function for decorating generating functions to
        convert them into producers.

        Typical usage example:
        @as_producer
        def gen_function(pro, *args, **kwargs):
            for arr in pro:
                yield np.fft(arr, axis=-1)
        #converts the generator function into  a producer for  other
        functions to consume.

    All other classes and functions of this module are not part of the 
    public API.
"""

import abc
import functools
import inspect
import itertools
import numpy as np
from collections.abc import Iterable, Sequence, Generator

from openseize.io.edf import Reader
from openseize.core import mixins

def producer(data, chunksize, axis, shape=None, mask=None, **kwargs):
    """Constructs an iterable that produces ndarrays of length chunksize
    along axis during iteration.

    This constructor returns an object that is capable of producing ndarrays
    or masked ndarrays during iteration from a single ndarray, a sequence of
    ndarrays, a file Reader instance (see io.bases.Reader), an ndarray 
    generating function, or a pre-existing producer of ndarrays. The 
    produced ndarrays from this object will have length chunksize along axis.

    Args:
        data:
            An object from which ndarrays will be produced from. Supported
            types are Reader instances, ndarrays, sequence of ndarrays, 
            generating functions yielding ndarrays, or a producer of 
            ndarrays. For sequences and generator functions it is
            required that each subarray has the same shape along all axes 
            except for the axis along which chunks will be produced. 
        chunksize: int
            The desired length along axis of each produced ndarray. 
        axis: int
            The sample axis of data that will be partitioned into 
            chunks of length chunksize.
        shape: tuple or None
            The combined shape of all ndarrays from this producer. This
            parameter is only required when object is a generating function
            and will be ignored otherwise.
        mask: 1-D boolean array
            A boolean describing which values of data along axis
            should by produced. Values that are True will be produced and
            values that are False will be ignored. If None (Default),
            producer will produce all values from object.
        kwargs: dict
            Keyword arguments specific to data type that ndarrays will be
            produced from. 
            For Reader instances, valid kwargs are channels
            and padvalue (see io.bases.Readers and io.edf.Reader)
            For generating functions, all the positional and keyword
            arguments must be passed to the function through these kwargs to
            avoid name collisions with the producer func arguments.

    Returns: An iterable of ndarrays of shape chunksize along axis.  
    """

    if isinstance(data, Producer):
        data.chunksize = int(chunksize)
        data.axis = axis
        result = data

    elif isinstance(data, Reader):
        result = ReaderProducer(data, chunksize, axis, **kwargs)

    elif inspect.isgeneratorfunction(data):
        result = GenProducer(data, chunksize, axis, shape, **kwargs)

    elif isinstance(data, np.ndarray):
        result = ArrayProducer(data, chunksize, axis, **kwargs)

    elif isinstance(data, Sequence):
        x = np.concatenate(data, axis=axis)
        result = ArrayProducer(x, chunksize, axis, **kwargs)

    else:
        msg = 'unproducible type: {}'
        raise TypeError(msg.format(type(data)))

    #apply mask if passed
    if mask is None:
        return result
    else:
        return MaskedProducer(result, mask, chunksize, axis, **kwargs)

def as_producer(func):
    """Decorator returning a Producer instance from a generating function.

    Producers are multi-transversal iterables (not iterators). Generators
    are 'one-shot' transversal objects. Therefore to support multiple data
    transversals, Producers can only be built from generating functions. 
    This decorater converts any generating function that accepts a producer
    as its first argument into a Producer instance.

    Returns: func
        A new function that creates a Producer instance in place of the
        decorated generating func.
    """

    # only decorate generator functions
    if not inspect.isgeneratorfunction(func):
        msg = 'as_producer requries a generating function not {}'
        raise TypeError(msg.format(type(func)))
    
    @functools.wraps(func)
    def decorated(pro, *args, **kwargs):
        """Returns a producer using values from generating func."""

        if not isinstance(pro, Producer):
            msg = ("First positional argument of decorated function"
                   " must be of type {} not {}")
            msg.format('Producer', type(pro))
   
        genfunc = functools.partial(func, pro, *args, **kwargs)
        return producer(genfunc, pro.chunksize, pro.axis, shape=pro.shape)

    return decorated


class Producer(Iterable, mixins.ViewInstance):
    """An ABC defining concrete and required methods of all Producers.

    Attrs:
        data: ndarray, sequence, memmap, file Reader, or generating func 
            An object that can return or yield ndarrays
        chunksize: int
            Number of samples along axis of data to yield per iteration.
        axis: int
            The sample axis of data along which data will be partitioned
            into chunks of chunksize and yielded per iteration.
        kwargs: dict
            All arguments required by data to produce ndarrays. These kwargs
            should include all required positional and keyword arguments
            required by data.

    As an ABC, this class cannot be instantiated. To create a producer
    instance use producer function constructor.
    """

    def __init__(self, data, chunksize, axis, **kwargs):
        """Concrete initializer for all Producer subclasses."""

        self.data = data
        self._chunksize = int(chunksize)
        self.axis = axis
        self.kwargs = kwargs
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


class ReaderProducer(Producer):
    """A Producer of ndarrays from an openseize file Reader instance.

    Attrs:
        see Producer attrs
        kwargs: dict
            Arguments passed to read method of a file reader instance.

    The data attribute in this Producer is a Reader instance
    """

    @property
    def shape(self):
        """Return the summed shape of all arrays in this Reader."""

        return self.data.shape

    def __iter__(self):
        """Builds an iterator yielding channels x chunksize shape arrays."""

        #make generators of start, stop samples & exhaust reader
        starts = itertools.count(start=0, step=self.chunksize)
        stops = itertools.count(start=self.chunksize, step=self.chunksize)
        for start, stop in zip(starts, stops): 
            arr = self.data.read(start, stop=stop, **self.kwargs)
            #if exhausted close reader and exit
            if arr.size == 0:
                break
            yield arr

    def close(self):
        """Closes this producer's file resource."""

        if hasattr(self.data, 'close'):
            self.data.close()


class ArrayProducer(Producer):
    """A Producer of ndarrays from an ndarray.

    Attrs:
        see Producer

    The data attribute in this Producer is an ndarray instance
    """

    @property
    def shape(self):
        """The cumulative shape of this Producers data array."""

        return self.data.shape

    def _slice(self, start, stop, step=None):
        """Builds a tuple of slice objs. between start and stop indexes."""

        slices = [slice(None)] * self.data.ndim
        slices[self.axis] = slice(start, stop, step)
        return tuple(slices)

    def __iter__(self):
        """Returns an iterator yielding ndarrays of chunksize along axis."""

        starts = range(0, self.data.shape[self.axis], self.chunksize)
        for t in itertools.zip_longest(starts, starts[1:], fillvalue=None):
            yield self.data[self._slice(*t)]


class GenProducer(Producer):
    """A Producer of ndarrays from a generating function of ndarrays.

    This iterable moves over a generator's yielded arrays collecting until
    chunksize shape is reached along axis. Once reached, it yields a 
    collected ndarray.

    Attrs:
        see Producer

    The data attribute in this Producer is a generating function
    """

    def __init__(self, data, chunksize, axis, shape, **kwargs):
        """Initialize this producer with additional required shape."""

        if shape is None:
            msg = 'A {} from a generating function requires a shape.'
            raise ValueError(msg.format('Producer'))

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
        for subarr in self.data(**self.kwargs):
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


class MaskedProducer(Producer):
    """A Producer of numpy arrays with values that have been filtered by
    a boolean mask.

    Attrs:
       see Producer
       mask: 1-D Bool 
            A bool mask to apply on all 1-D slices of producer's array 
            oriented along axis (see np.take). Iteration of arrays stops
            as soon as mask or Producer runs out of values to yield.

    The data attribute in this Producer is a Producer instance
    """

    def __init__(self, pro, mask, chunksize, axis, **kwargs):
        """Initialize this Producer with a boolean array mask."""

        super().__init__(pro, chunksize, axis, **kwargs)
        self.mask = producer(mask, chunksize, axis=0, *kwargs)

    @property
    def shape(self):
        """The cumulative shape of all arrays in this producer."""

        result = list(self.data.shape[axis])
        #iteration stops when producer or mask runs out
        result[self.axis] = min(self.data.shape[axis], 
                                np.count_nonzero(self.mask))
        return tuple(shape)

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


if __name__ == '__main__':

    """
    def gfunc(count):
        #A generating function to play with.
        
        np.random.seed(0)
        arrs = [np.random.random((4, 7)) for _ in range(count)]
        for arr in arrs:
            yield arr

    pro = producer(gfunc, chunksize=10, axis=-1, shape=(4, 7*10), count=4)
    """

    fp = ('/home/matt/python/nri/data/rett_eeg/dbs_treated/edf/'
          '5872_Left_group A-D.edf')
    from openseize.io.edf import Reader
    reader = Reader(fp)
    pro = producer(reader, chunksize=10000, axis=-1, channels=[0,2])
 
