"""Tools for building Pipelines, callables composed of subcallables that
sequentially act on a producer when the Pipeline instance is called.

This module includes:

    Pipeline:
        A callable that executes a sequence of functions on any input.
"""

from copy import copy
from functools import partial
import inspect
from typing import Callable, List


class Pipeline:
    """A callable that executes a chain of callables each accepting as input the
    output of the previous callable.

    Pipelines are compositions of complex sequences of DSP operations that
    will execute on producers, ndarrays, or other data structures when needed.
    They support multiprocessing and are self documenting since the functions
    executed and their parameters are stored within a pipeline.

    Attrs:
        callers:
            A sequence of callables to execute.

    Examples:
    >>> import numpy as np
    >>> from openseize.core.producer import producer, Producer
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
    >>> # assert the pipeline contains the notch function
    >>> notch in transformer
    True
    """

    def __init__(self) -> None:
        """Initialize this Pipeline with an empty list of callables."""

        self.callers: List[Callable] = []

    def validate(self, caller: Callable, **kwargs) -> None:
        """Validates that a partial of caller constructed with kwargs has
        exactly on free argument."""

        sig = inspect.signature(caller)
        bound = sig.bind_partial(**kwargs)
        bound.apply_defaults()
        unbound_cnt = len(sig.parameters) - len(bound.arguments)
        if unbound_cnt > 1:
            msg = ('Pipeline callers must have exactly one unbound argument.'
                   f' {caller.__name__} has {unbound_cnt} unbound arguments.')
            raise TypeError(msg)

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

        self.validate(caller, **kwargs)
        frozen = partial(caller, **kwargs)
        self.callers.append(frozen)

    def __contains__(self, caller: Callable) -> bool:
        """Returns True if func with name is in this Pipeline's callables.

        Args:
            caller:
                A callable to find in this pipelines partial callers.

        Returns:
            True if caller is in callers and False otherwise.
        """

        return caller in [caller.func for caller in self.callers]

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
