"""

"""

import abc
import functools
import inspect
import itertools
import numpy as np
from collections.abc import Iterable, Sequence, Generator

from openseize.io.edf import Reader
from openseize.core import mixins

def producer(obj, chunksize, axis, shape=None, mask=None, **kwargs):
    """Constructs an iterable that produces ndarrays of length chunksize
    along axis during iteration.

    This constructor returns an object that is capable of producing ndarrays
    or masked ndarrays during iteration from a single ndarray, a sequence of
    ndarrays, a file Reader instance (see io.bases.Reader), an ndarray 
    generating function, or a pre-existing producer of ndarrays. The 
    produced ndarrays from this object will have length chunksize along axis.

    Args:
        obj:
            An object from which ndarrays will be produced from. Supported
            object types are Reader instances, ndarrays, sequence of
            ndarrays, generating functions yielding ndarrays, or a producer
            of ndarrays. For sequences and generator functions it is
            required that each subarray has the same shape along all axes 
            except for the axis along which chunks will be produced. 
        chunksize: int
            The desired length along axis of each produced ndarray. 
        axis: int
            The axis of data in obj that will be partitioned into chunks
            of length chunksize.
        shape: tuple or None
            The combined shape of all ndarrays from this producer. This
            parameter is only required when object is a generating function
            and will be ignored otherwise.
        mask: 1-D boolean array
            A boolean describing which values of data in obj along axis
            should by produced. Values that are True will be produced and
            values that are False will be ignored. If None (Default),
            producer will produce all values from object.
        kwargs: dict
            Keyword arguments specific to obj type that ndarrays will be
            produced from. 
            For Reader instances, valid kwargs are channels
            and padvalue (see io.bases.Readers and io.edf.Reader)
            For generating functions, kwargs are the functions parameters.

    Returns: An iterable of ndarrays of shape chunksize along axis.  
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
        data = np.concatenate(obj, axis=axis)
        result = ArrayProducer(data, chunksize, axis, **kwargs)

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

    # FIXME Several issues with this decorator
    # 1. instead of requiring 1st param to be a producer, we could pass the
    #    params needed to build a producer (chunksize, axis, shape) as
    #    decorator parameters
    # 2. It is unclear in docs why this is needed. It avoids clients having
    #    to form partial funcs to build producers. Remember producers can
    #    not be built from generators only generating funcs.

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
        
        # FIXME since all args are frozen, producer does not need kwargs
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
        kwargs:                                passed to obj being produced
        from (valid only for generating  functions)

    As an ABC, this class cannot be instantiated. To create a producer
    instance use producer function.
    """

    def __init__(self, data, chunksize, axis, **kwargs):
        """Concrete initializer for all Collector subclasses."""

        self.data = data
        self._chunksize = int(chunksize)
        self.axis = axis
        self.kwargs = kwargs
        # FIXME what does this mean?
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

    #FIXME verify not needed
    """
    def __init__(self, data, chunksize, axis, channels=None, 
                 padvalue=np.NaN):
        #Initialize Producer with additional channels and padvalues.

        super().__init__(data, chunksize, axis, channels=channels,
                         padvalue=padvalue)
        if not channels:
            self.channels = self.data.header.channels
    """

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
            #arr = self.data.read(start, stop, self.channels, self.padvalue)
            # FIXME verify kwargs passed this way work
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


if __name__ == '__main__':

    def gfunc(count):
        """A generating function to play with."""
        
        np.random.seed(0)
        arrs = [np.random.random((4, 7)) for _ in range(count)]
        for arr in arrs:
            yield arr

    pro = producer(gfunc, chunksize=10, axis=-1, shape=(4, 7*10), count=4)
