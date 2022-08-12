import numpy as np


class FIFOArray:
    """A first-in-first-out queue-like data structure for collecting 
    ndarrays into chunksize subarrays.

    Attrs:
        queue: ndarray
            An ndarray formed by the concatenation of arrays supplied by the
            'put' method along axis.
        chunksize: int
            The size of each ndarray to be dequeued by the 'get' method.
        axis: int
            The axis along which ndarrays will be collected. This is usually
            the sample axis of the putted ndarrays.
    """

    def __init__(self, chunksize, axis):
        """Initialize this FIFOArray with an empty queue."""

        self.queue = np.array([])
        self.chunksize = chunksize
        self.axis = axis

    def qsize(self):
        """Returns the shape of the queue along axis."""

        return 0 if self.empty() else self.queue.shape[self.axis]

    def empty(self):
        """Returns True if the queue is empty and False otherwise."""

        return True if self.queue.size == 0 else False

    def full(self):
        """Returns True if the size of the queue is at least chunksize."""

        return True if self.qsize() >= self.chunksize else False

    def put(self, x):
        """Appends an ndarray to this queue.

        Args:
            x: ndarray
                If the queue is not empty, this array must have the same
                dimensions as the queue on all axes except axis.
        """

        if self.empty():
            self.queue = x
        else:
            self.queue = np.concatenate((self.queue, x), axis=self.axis)

    def get(self):
        """Pops an array of size chunksize along axis from this queue."""

        result, self.queue = np.split(self.queue, [self.chunksize],
                                      axis=self.axis)
        return result

