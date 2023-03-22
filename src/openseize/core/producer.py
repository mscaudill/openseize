"""Producers are the heart of Openseize. They are iterables that can produce
values from a variety of data sources including:
```
    - Sequences
    - Numpy ndarrays
    - Generating functions
    - Openseize Reader instances
    - Other Producers
```

All DSP algorithms in Openseize can accept and return a producer. This
allows data that is too large to be stored to an in-memory numpy array to be
analyzed. This module contain the producer constructing function. It is the
only publicly available method of this module.

Examples:

    >>> # Build a producer from an EDF file reader
    >>> from openseize.demos import paths
    >>> filepath = paths.locate('recording_001.edf')
    >>> from openseize.file_io.edf import Reader
    >>> reader = Reader(filepath)
    >>> # build a producer that produces 100k samples chunks of this file
    >>> pro = producer(reader, chunksize=10e3, axis=-1)
    >>> pro.shape # print the producers shape
    (4, 18875000)
    >>> # print the shape of each arr in the producer
    >>> for idx, arr in enumerate(pro):
    ...     msg = 'Array num. {} has shape {}'
    ...     print(msg.format(idx, arr.shape))
    >>> # Build a producer from a numpy array with samples on 0th
    >>> x = np.random.random((100000, 5))
    >>> xpro = producer(x, chunksize=10e3, axis=0)
    >>> for arr in xpro:
    ...     print(arr.shape)
"""

from abc import abstractmethod
from collections import abc
import functools
import inspect
import itertools
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from openseize.core import mixins
from openseize.core import resources
from openseize.core.queues import FIFOArray
from openseize.file_io.edf import Reader


def producer(data: Union[npt.NDArray, Iterable[npt.NDArray], Reader,
                         Callable, 'Producer'],
             chunksize: int,
             axis: int,
             shape: Optional[Sequence[int]] = None,
             mask: Optional[npt.NDArray[np.bool_]] = None,
             **kwargs,
) -> 'Producer':
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
            types are Reader instances, ndarrays, sequences, generating
            functions yielding ndarrays, or other Producers.  For sequences
            and generator functions it is required that each subarray has
            the same shape along all axes except for the axis along which
            chunks will be produced.
        chunksize:
            The desired length along axis of each produced ndarray.
        axis:
            The sample axis of data that will be partitioned into
            chunks of length chunksize.
        shape:
            The combined shape of all ndarrays from this producer. This
            parameter is only required when object is a generating function
            and will be ignored otherwise.
        mask:
            A boolean describing which values of data along axis
            should by produced. Values that are True will be produced and
            values that are False will be ignored. If None (Default),
            producer will produce all values from object.
        kwargs:
            Keyword arguments specific to data type that ndarrays will be
            produced from. For Reader instances, valid kwargs are padvalue
            (see io.bases.Readers and io.edf.Reader) For generating
            functions, all the positional and keyword arguments must be
            passed to the function through these kwargs to avoid name
            collisions with the producer func arguments.

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

    elif isinstance(data, abc.Sequence):
        x = np.concatenate(data, axis)
        result = ArrayProducer(x, chunksize, axis, **kwargs)

    else:
        msg = 'unproducible type: {}'
        raise TypeError(msg.format(type(data)))

    # apply mask if passed
    if mask is None:
        return result

    return MaskedProducer(result, mask, chunksize, axis, **kwargs)


def as_producer(func):
    """Decorator returning a Producer instance from a generating function.

    Producers are multi-transversal iterables (not iterators). Generators
    are 'one-shot' transversal objects. Therefore to support multiple data
    transversals, Producers can be built from generating functions.
    This decorater converts any generating function that accepts a producer
    as its first argument into a Producer instance.

    Caveats:
        This decorator assumes that the generating function yields arrays
        whose cumulative shape matches the input producers shape. If this is
        False, it is advised to construct the producer from the producer
        function passing an explicit shape argument.
            >>> producer(genfunc, chunksize, axis, shape, **kwargs)

        The chunksize of the resultant producer will match the chunksize of
        the input producer to the generating function

    Returns: func
        A func. that creates a Producer instance from a generating func.
    """

    # only decorate generator functions
    if not inspect.isgeneratorfunction(func):
        msg = 'as_producer requires a generating function not {}'
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


def pad_producer(pro, pad, value):
    """Pads the edges of a producer along its axis.

    Args:
        pro: producer of ndarrys
            The producer that is to be padded.
        pad: 2-el sequence of ints or single int
            The number of pads to apply before the 0th and after the last
            index of the producer's axis. If int, pads will be added to
            both.
        value: float
            The constant value to pad the producer with. Defaults to zero.

    Returns: A producer yielding ndarrays.
    """

    #convert int pad to seq. of pads & place along axis of pads
    pads = [pad, pad] if isinstance(pad, int) else pad

    def genfunc():

        left_shape, right_shape = list(pro.shape), list(pro.shape)
        left_shape[pro.axis] = pads[0]
        right_shape[pro.axis] = pads[1]
        left = value * np.ones(left_shape)
        right = value * np.ones(right_shape)

        yield left

        for arr in pro:
            yield arr

        yield right

    # compute new shape
    shape = list(pro.shape)
    shape[pro.axis] = pro.shape[pro.axis] + sum(pads)

    return producer(genfunc, pro.chunksize, pro.axis, shape=shape)


class Producer(abc.Iterable, mixins.ViewInstance):
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

    @property
    def chunksize(self):
        """Returns the chunksize of this Producer."""

        return self._chunksize

    @chunksize.setter
    def chunksize(self, value):
        """Sets this Producer's chunksize attr ensuring it is int type."""

        self._chunksize = int(value)

    @property
    @abstractmethod
    def shape(self):
        """Returns the shape of this producers data attr."""

    def to_array(self, dtype=float):
        """Assign this Producer to an ndarray by concatenation along axis.

        Args:
            dtype: numpy datatype
                The datatype of each sample in this Producer. Default is
                float64.
        """

        resource_result  = resources.assignable_array(self.shape, dtype)
        assignable, allowable, required = resource_result

        if not assignable:
            a, b = np.round(np.array([required, allowable]) / 1e9, 1)
            msg = 'Producer will consume {} GB but only {} GB are available'
            raise MemoryError(msg.format(a, b))

        return np.concatenate(list(self), axis=self.axis)


class ReaderProducer(Producer):
    """A Producer of ndarrays from an openseize file Reader instance.

    Attrs:
        see Producer attrs
        kwargs: dict
            Arguments passed to read method of a file reader instance.

    Notes:
        The data attribute of this Producer is a closed Reader instance which is
        opened during iteration. This allows producers to be serialized for
        concurrent processing. It also means that when a ReaderProducer is
        created all other producers using the same file will stop producing.
    """

    def __init__(self, data, chunksize, axis, **kwargs):
        """Initialize this Producer with a closed 'data' Reader instance."""

        super().__init__(data, chunksize, axis, **kwargs)
        self.data.close()

    @property
    def shape(self):
        """Return the summed shape of all arrays in this Reader."""

        return self.data.shape

    def __iter__(self):
        """Builds an iterator yielding channels x chunksize shape arrays."""

        # Open the data reader
        self.data.open(**self.kwargs)
        # make generators of start, stop samples & exhaust reader
        starts = itertools.count(start=0, step=self.chunksize)
        stops = itertools.count(start=self.chunksize, step=self.chunksize)

        for start, stop in zip(starts, stops):
            arr = self.data.read(start, stop=stop)
            # if exhausted close reader and exit
            if arr.size == 0:
                break

            yield arr


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
        shape: tuple
            The combined shape of all arrays in data generating function.
        kwargs: dict
            Any valid keyword argument for the data generating function.

    The data attribute in this Producer is a generating function
    """

    def __init__(self, data, chunksize, axis, shape, **kwargs):
        """Initialize this producer with additional required shape."""

        if shape is None:
            msg = 'A {} from a generating function requires a shape.'
            raise ValueError(msg.format('Producer'))

        super().__init__(data, chunksize, axis, **kwargs)
        self._shape = tuple(shape)

    @property
    def shape(self):
        """Returns the summed shape of arrays in this Producer."""

        return self._shape

    def __iter__(self):
        """Returns an iterator yielding ndarrays of chunksize along axis.

        Since the shape of the arrays yielded from the generating function
        (i.e. data) may be very small compared to chunksize, we store
        generated arrays to a temporary array to reduce the number of fifo
        'put' calls (see note in FIFOArray.put).
        """

        # collector will fetch chunksize array for each 'get' call
        collector = FIFOArray(self.chunksize, self.axis)

        # make tmp array to hold generated subarrs
        tmp = []
        tmp_size = 0
        for subarr in self.data(**self.kwargs):

            tmp.append(subarr)
            tmp_size += subarr.shape[self.axis]

            # if tmp exceeds chunksize put in collector
            if tmp_size >= self.chunksize:
                arr = np.concatenate(tmp, axis=self.axis)
                collector.put(arr)

                # fetch chunksize till not full
                while collector.full():
                    yield collector.get()

                # place leftover back into tmp and empty collector
                tmp = [collector.queue]
                tmp_size = collector.qsize()
                collector.queue = np.array([])

            else:

                # append to tmp again
                continue

        # else runs after normal loop exit -- required here
        else: #pylint: disable=useless-else-on-loop

            # yield whatever is left in tmp (its below chunksize)
            remaining = np.concatenate(tmp, axis=self.axis)
            if remaining.size > 0:
                yield remaining


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
        self.mask = producer(mask, chunksize, axis=0)

    @property
    def shape(self):
        """The cumulative shape of all arrays in this producer."""

        result = list(self.data.shape)
        included  = np.count_nonzero(self.mask.to_array(dtype=bool))
        #iteration stops when producer or mask runs out
        result[self.axis] = min(self.data.shape[self.axis], included)

        return tuple(result)

    @property
    def chunksize(self):
        """Returns the chunksize of this MaskedProducer."""

        return self.data.chunksize

    @chunksize.setter
    def chunksize(self, value):
        """On change, set chunksize for both producer and mask."""

        self.data.chunksize = int(value)
        self.mask.chunksize = int(value)

    def __iter__(self):
        """Returns an iterator of boolean masked numpy arrays along axis."""

        collector = FIFOArray(self.chunksize, self.axis)
        for arr, maskarr in zip(self.data, self.mask):
            filtered = np.take(arr, np.flatnonzero(maskarr), axis=self.axis)
            collector.put(filtered)

            while collector.full():

                yield collector.get()

        # else runs after normal loop exit -- required here
        else: #pylint: disable=useless-else-on-loop

            if collector.qsize() > 0:

                yield collector.get()
