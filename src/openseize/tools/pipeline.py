"""A class for building sequences of callables (aka pipelines) that delay their
actions until their call method is invoked.

This module includes:
    
    Pipeline:
        A callable that executes a sequence of functions on any input.
"""

from copy import copy
from functools import partial
from typing import Any, Callable, List


class Pipeline:
    """A callable that executes a chain of callables each accepting as input the
    output of the previous callable.

    Pipelines are powerful. They compose complex sequences of operations that
    will execute on producers, ndarrays, or other data structures when needed.
    They support multiprocessing and are self documenting since the functions
    executed and their parameters are stored within a pipeline.

    Attrs:
        callers:
            A sequence of callables to execute.

    Examples:
    >>> import numpy as np
    >>> from openseize import producer
    >>> from openseize.filtering.iir import Notch
    >>> from openseize.resampling.resampling import downsample
    >>> # make a producer from a random 2D array
    >>> arr = np.random.random((4, 123000))
    >>> pro = producer(arr, chunksize=5000, axis=-1)
    >>> # build a 60Hz notch filter assuming fs = 1000 Hz
    >>> notch = Notch(60, width=6, fs=1000)
    >>> # build a transforming Pipeline and add the notch
    >>> transformer = Pipeline()
    >>> transformer.append(notch, chunksize=1000, axis=-1)
    >>> # add a 2-fold downsampler to this transformer
    >>> transformer.append(downsample, M=2, fs=1000, chunksize=1000, axis=-1)
    >>> # call the transformer on the producer
    >>> result = transformer(pro)
    >>> # assert that we have a producer and it has the expected shape
    >>> isinstance(result, Producer)
    True
    >>> result.shape
    (4, 61500)
    """

    def __init__(self) -> None:
        """Initialize this Pipeline with an empty list of callables."""

        self.callers: List[Callable] = []

    def append(self, caller: Callable, **kwargs) -> None:
        """Append a callable to this Pipeline.
        
        Args:
            caller:
                A callable to append to this pipeline's callables.
            kwargs:
                All arguments with the exception of the data argument.

        Notes:
            The kwargs must supply all required arguments of the caller except
            for the data argument. A partial function will be created and
            will raise a TypeError during the __call__ method if more than 1 
            unbound argument is present.
        """

        frozen = partial(caller, **kwargs)
        self.callers.append(frozen)

    def __contains__(self, name: str) -> bool:
        """Returns True if func with name is in the list of this Pipeline's
        callables.

        Args:
            name:
                The name of a function to look up in this Pipeline.

        Returns:
            True if named function is in callers and False otherwise.
        """

        names = [caller.__name__ for caller in self.callers]
        return name in names

    def __call__(self, data):
        """Apply this Pipeline's callables to an initial data argument.

        Args:
            data:
                Input data of any type but typically an openseize producer 
                or numpy ndarray.

        Returns:
            Any value depending on the last callable in callers.
        """

        res = copy(data)
        for caller in self.callers:
            res = caller(res)
        return res
