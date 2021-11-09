import numpy as np
import abc
import functools
import inspect
import itertools
from collections.abc import Iterable, Sequence, Generator

from openseize.io.readers import Reader
from openseize.types import mixins

def producer(obj, chunksize, axis, shape=None, mask=None, **kwargs):
    """Returns a maskable iterable of ndarrays of shape chunksize along 
    axis from objects of type: ndarray, Sequence, openseize Reader, or
    generating functions yielding ndarrays.

    Args:
        obj:                    A Reader instance, ndarray, Sequence of
                                ndarrays, Producer instance or generating 
                                function yielding ndarrays.

        chunksize (int):        len of each ndarray yielded by this iterable
                                along axis parameter.

        axis (int):             Axis of obj to partition into chunks of 
                                chunksize.

        shape (tuple):          Tuple of ints describing objects total size 
                                with samples expected along axis. This 
                                parameter is only requried when obj is
                                a generating function and ignored otherwise.
        
        mask (1-D bool):        boolean array to mask producer outputs along
                                axis. Values of obj at True mask indices 
                                will be yielded and False will be ignored. 
                                If mask len does not match obj len along 
                                axis, The producer yields subarrays upto the 
                                shorter of mask and obj.
        
        kwargs:                 kwargs specific to a Producer subtype. E.g.
                                channels & padvalue are valid kwargs for a 
                                ReaderProducer
        
    Returns: a Producer iterable   
    """

    if isinstance(obj, Producer):
        obj.chunksize = int(chunksize)
        obj.axis = axis
        result = obj

    elif isinstance(obj, Reader):
        result = ReaderProducer(obj, chunksize, axis, **kwargs)

    elif inspect.isgeneratorfunction(obj):
        result = GenProducer(obj, chunksize, axis, shape, **kwargs)

    elif isinstance(obj, np.ndarray):
        result = ArrayProducer(obj, chunksize, axis, **kwargs)

    elif isinstance(obj, Sequence):
        data = np.array(obj)
        result = ArrayProducer(obj, chunksize, axis, **kwargs)

    else:
        msg = 'unproducible type: {}'
        raise TypeError(msg.format(type(obj)))

    #apply mask if passed
    if mask is None:
        return result
    else:
        return MaskedProducer(result, mask, chunksize, axis, **kwargs)

def as_producer(func):
    """Decorator that returns a Producer from a generating function.

    Producing from a generator requires the original generating function
    since iteration by a Producer exhaust single use generators. This
    decorator can be used to decorate any generating function recasting it
    as a Producer that can regenerate from a generating function as many
    times as needed.

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


class ReaderProducer(Producer):
    """Producer of ndarrays from an openseize file Reader instance.

    Attrs:
        see Producer
        channels (seq):                 seq of channel indices to yield
        padvalue (float):               A reader may be reading a file with
                                        signals that were sampled at 
                                        different sampling rates and 
                                        therefore have different lengths. 
                                        In order to yield a rectangular 
                                        array on each batch fetched, this 
                                        padvalue will be appended so all
                                        signals are the same lenght as the
                                        longest signal in the file. The
                                        default is to pad np.NaN values.

    The data attribute in this Producer is a Reader instance
    """

    def __init__(self, data, chunksize, axis, channels=None, 
                 padvalue=np.NaN):
        """Initialize Producer with additional channels and padvalues."""

        super().__init__(data, chunksize, axis, channels=channels,
                         padvalue=padvalue)
        if not channels:
            self.channels = self.data.header.channels

    @property
    def shape(self):
        """Return the summed shape of all arrays in this Reader."""

        return self.data.shape

    def __iter__(self):
        """Returns an iterator yielding arrays of shape channels x chunksize
        from a reader instance."""

        #make generators of start, stop samples & exhaust reader
        starts = itertools.count(start=0, step=self.chunksize)
        stops = itertools.count(start=self.chunksize, step=self.chunksize)
        for start, stop in zip(starts, stops): 
            arr = self.data.read(start, stop, self.channels, self.padvalue)
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


class MaskedProducer(Producer):
    """A Producer of numpy arrays with values that have been filtered by
    a boolean mask.

    Attrs:
       see Producer
       mask (1-D Bool):         a 1-D boolean array of masked values. The
                                length of the mask does not have to match 
                                the length of the producer but 
                                MaskedProducer will stop producing as soon
                                as the producer or mask raise StopIteration.
    
    The data attribute in this Producer is a Producer instance

    Note: The bool mask is applied on all 1-D slices of producer's array
    oriented along axis (see np.take).
    """

    def __init__(self, pro, mask, chunksize, axis, **kwargs):
        """Initialize this Producer with a boolean array mask."""

        super().__init__(pro, chunksize, axis, **kwargs)
        self.mask = producer(mask, chunksize, axis=0, *kwargs)

    @property
    def shape(self):
        """Return the shape of this Producers data attr."""

        return self.data.shape

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
