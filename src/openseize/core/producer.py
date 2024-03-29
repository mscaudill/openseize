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
from itertools import zip_longest
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
            Keyword arguments are specific to data argument type:
            - Reader:
                padvalue:
                    see reader.read method
                start:
                    The start sample to begin data production along axis.
                stop:
                    The stop sample to halt data production along axis.
            - Generating Function:
                All positional and keyword arguments to the Gen. func. must be passed
                through these kwargs to avoid name collisions with the producer
                func arguments.
            - Arrays:
                The kwargs are ignored.
            - Sequences:
                The kwargs are ignored

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

    def to_array(self, dtype=float, limit=None):
        """Assign this Producer to an ndarray by concatenation along axis.

        Args:
            dtype: numpy datatype
                The datatype of each sample in this Producer. Default is
                float64.
            limit: int
                The maximum memory to consume in assigning this producer. If
                None, the limit will be all available memory.
        """

        if resources.assignable(self.shape, dtype, limit=limit):
            return np.concatenate(list(self), axis=self.axis)


class ReaderProducer(Producer):
    """A Producer of ndarrays from an openseize file Reader instance.

    Attrs:
        Producer Attrs
        start:
            The start sample along production axis at which data production
            begins.
        stop:
            The stop sample along production axis at which data production
            stops.
        kwargs:

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

        # Pop the start and stop from kwargs
        a = self.kwargs.pop('start', 0)
        b = self.kwargs.pop('stop', self.data.shape[axis])
        self.start, self.stop, _ = slice(a, b).indices(data.shape[axis])

        # close for serialization
        self.data.close()

    @property
    def shape(self):
        """Return the summed shape of all arrays in this Reader."""

        s = list(self.data.shape)
        s[self.axis] = self.stop - self.start
        return tuple(s)

    def __iter__(self):
        """Builds an iterator yielding channels x chunksize shape arrays."""

        # Open the data reader
        self.data.open()

        starts = np.arange(self.start, self.stop, self.chunksize)
        for a, b in zip_longest(starts, starts[1:], fillvalue=self.stop):
            yield self.data.read(a, b, **self.kwargs)


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

        for t in zip_longest(starts, starts[1:], fillvalue=None):
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

            if not np.any(maskarr):
                continue

            filtered = np.take(arr, np.flatnonzero(maskarr), axis=self.axis)
            collector.put(filtered)

            while collector.full():

                yield collector.get()

        # else runs after normal loop exit -- required here
        else: #pylint: disable=useless-else-on-loop

            if collector.qsize() > 0:

                yield collector.get()
