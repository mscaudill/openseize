"""A collection of tools to manipulate the size, shape or values produced by
a producer.
"""

from collections.abc import Iterator
from functools import partial
from itertools import zip_longest
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from openseize import producer
from openseize.core import arraytools
from openseize.core.producer import Producer


def pad(
    pro: Producer,
    amt: Union[int, Tuple[int, int]],
    axis: int,
    value: Optional[float] = 0,
) -> Producer:
    """Pads the edges of a producer along single axis with a constant value.

    Args:
        pro:
            A producer of ndarrays whose edges along axis are to be padded.
        amt:
            The number of pads to apply before the 0th element & after the
            last element along axis. If int, amt number of values will be
            prepended & appended to axis.
        axis:
            The axis of produced values along which to pad.
        value: float
            The constant value to pad the producer with. Defaults to zero.

    Examples:
        >>> x = np.arange(1000).reshape(4, 250)
        >>> pro = producer(x, chunksize=100, axis=-1)
        >>> padded_pro = pad(pro, [3, 10], axis=-1)
        >>> padded_pro.shape
        (4, 263)
        >>> np.allclose(np.pad(x, [(0,0), (3, 10)]), padded_pro.to_array())
        True

    Returns:
        A new producer padded with value along axis.

    Notes:
        This padding is less sophisticated than numpy as openseize only allows
        constant pre and post padding. Future versions will likely improve this.
    """

    amts = (amt, amt) if isinstance(amt, int) else tuple(amt)

    # dispatch to generating function based on whether pad is along pro.axis
    if arraytools.normalize_axis(axis, pro.ndim) == pro.axis:
        genfunc = _production_axis_padder
    else:
        genfunc = _other_axis_padder

    # build a partial generating function and compute the return pros shape
    func = partial(genfunc, pro, amts, axis, value)
    new_shape = list(pro.shape)
    new_shape[axis] = pro.shape[axis] + sum(amts)

    return producer(func, pro.chunksize, pro.axis, shape=new_shape)


def _production_axis_padder(pro, amt, axis, value):
    """A generating function that pads a producer along its axis with value.

    Padding a producer along its production axis only changes the first and last
    produced arrays. For argument definitions see pad.
    """

    left_shape, right_shape = list(pro.shape), list(pro.shape)
    left_shape[axis] = amt[0]
    right_shape[axis] = amt[1]

    # create the arrays to pad left and right along axis
    left, right = [value * np.ones(s) for s in (left_shape, right_shape)]

    yield left

    yield from pro

    yield right


def _other_axis_padder(pro, amt, axis, value):
    """A generating func. that pads a producer along any non-production axis.

    Padding a producer along a non-production axis changes the shape of all
    produced arrays.
    """

    for arr in pro:
        yield arraytools.pad_along_axis(arr, amt, axis, constant_values=value)


def expand_dims(pro: Producer, axis: Union[int, Tuple] = 0) -> Producer:
    """Expands a producer's shape by inserting a new axis at axis position.

    Args:
        producer:
            A producer of ndarrays.
        axis:
            The position in the expanded axes where the axis or axes are placed.

    Examples:
        >>> data = np.random.random((102344,))
        >>> pro = producer(data, chunksize=100, axis=-1)
        >>> print(pro.shape)
        (102344,)
        >>> print(pro.axis)
        0
        >>> expanded = expand_dims(pro, axis=(0, -1))
        >>> print(expanded.shape)
        (1, 102344, 1)
        >>> # take note the producing axis changes too!
        >>> print(expanded.axis)
        1

    Returns:
        A new producer with expanded dimensions.

    Notes:
        In contrast with numpy's expand_dims, this function must expand the
        produced array dims and track where the producing axis ends up. Callers
        should be aware that inserting new axes into a producer may change the
        production axis.
    """

    # normalize the axis to insert and the producer's axis
    axes = (axis,) if isinstance(axis, int) else axis

    # calculate out ndims, initialize new shape and normalize inserts
    new_ndim = len(pro.shape) + len(axes)
    new_shape = np.ones(new_ndim, dtype=int)
    inserts = [arraytools.normalize_axis(ax, new_ndim) for ax in axes]

    # find indices of new_shape where we will insert producer's shape
    complements = sorted(set(range(new_ndim)).difference(inserts))

    # set the new axis and insert producer's shape into new shape
    new_axis = complements[pro.axis]

    for idx, comp in enumerate(complements):

        new_shape[comp] = pro.shape[idx]

    new_shape = [int(s) for s in new_shape]
    func = partial(_expand_gen, pro, axes)
    return producer(func, pro.chunksize, new_axis, tuple(new_shape))


def _expand_gen(pro, axes):
    """A generating function that expands the dims of each produced array
    in a producer.

    Args:
        pro:
            A producer of ndarrays.
        axes:
            A tuple of axes to insert.

    Yields:
        Arrays with expanded dims.
    """

    for arr in pro:
        yield np.expand_dims(arr, axes)


def multiply_along_axis(
    pro: Producer,
    arr: npt.NDArray,
    axis: int,
) -> Producer:
    """Multiplies produced arrays by a 1-D array along a single axis.

    Args:
        pro:
            A producer of ndarrays to be multiplied along axis.
        arr:
            A 1-D array whose length must match producers length along a single
            axis.
        axis:
            The axis along which to multiply. This function supports
            multiplication along any single axis including the production axis.

    Examples:
        >>> x = np.arange(10000).reshape(2, 4, 1250)
        >>> pro = producer(x, chunksize=100, axis=-1)
        >>> arr = np.array([0, -1, 1, 0]) #1D array to multiply by
        >>> multiplied = multiply_along_axis(pro, arr, axis=1)
        >>> y = multiplied.to_array()
        >>> np.allclose(x * arr.reshape(1, 4, 1), y)
        True

    Returns:
        A new producer of arrays the same shape as the input producer.
    """

    arr = np.array(arr)
    if arr.ndim > 1:
        raise ValueError("Dimensions of multiplier arr must be exactly 1.")

    # ensure the arr shape matches the producers shape along axis
    if len(arr) != pro.shape[axis]:
        msg = "operands could not be broadcast together with shapes {} {}"
        raise ValueError(msg.format(pro.shape, arr.shape))

    # reshape the input array to be broadcastable with produced arrays
    ndims = len(pro.shape)
    shape = np.ones(ndims, dtype=int)
    shape[axis] = len(arr)
    x = arr.reshape(shape)  # type: Union[npt.NDArray, Producer]

    # if multiplying along pro axis convert arr 'x' to producer
    if arraytools.normalize_axis(axis, pro.ndim) == pro.axis:
        x = producer(x, chunksize=pro.chunksize, axis=pro.axis)

    func = partial(_multiply_gen, pro, x)
    return producer(func, chunksize=pro.chunksize, axis=pro.axis, shape=pro.shape)


def _multiply_gen(pro, multiplier):
    """A generating helper function that multiplies produced arrays by an
    ndarray or producer of ndarrays.

    This helper function is a generating function (not a producer) and is not
    intended to be called externally. It assumes that multipliers shape is
    broadcastable to producers shape.

    Args:
        pro:
            A producer of ndarrays.
        multiplier:
            An ndarray or a producer of ndarrays. The number of dims of this
            object must match the dims of pro and have shape of 1 along all axes
            except 1 axis whose length must equal the length of the producer
            along this axis.

    Yields:
        The element-wise product of each produced array with multiplier.
    """

    # non-production axis multiplication factors
    factors = zip_longest(pro, multiplier, fillvalue=multiplier)

    # production axis multiplication factors
    if isinstance(multiplier, Producer):
        factors = zip(pro, multiplier)

    for arr, mult in factors:
        yield arr * mult


def slice_along_axis(
    pro: Producer,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: Optional[int] = None,
    axis: int = -1,
) -> Producer:
    """Returns a producer producing values between start and stop in step
    increments along axis.

    Args:
        pro:
            A producer instance to slice along axis.
        start:
            The start index of the slice along axis. If None, slice will start
            at 0.
        stop:
            The stop index of the slice along axis. If None slice will extend to
            last element(s) of producer along axis.
        step:
            The size of index steps between start and stop of slice.
        axis:
            The axis of the producer to be sliced.

    Examples:
        >>> x = np.random.random((4,10000))
        >>> pro = producer(x, chunksize=1000, axis=-1)
        >>> sliced_pro = slice_along_axis(pro, 100, 200)
        >>> np.allclose(x[:,100:200], sliced_pro.to_array())
        True

    Returns:
        A producer of ndarrays.
    """

    # get start, stop, step indices for the slicing axis
    start, stop, step = slice(start, stop, step).indices(pro.shape[axis])

    if arraytools.normalize_axis(axis, pro.ndim) == pro.axis:
        # slicing along production axis is just masking
        mask = np.zeros(pro.shape[axis], dtype=bool)
        mask[start:stop:step] = True
        return producer(pro, pro.chunksize, pro.axis, mask=mask)

    # slicing along non-production axis changes shape of produced arrays
    new_shape = list(pro.shape)
    new_shape[axis] = (stop - start) // step
    func = partial(_slice_along_gen, pro, start, stop, step, axis)
    return producer(func, pro.chunksize, pro.axis, shape=new_shape)


def _slice_along_gen(pro, start, stop, step, axis):
    """A generating helper function for slicing a producer along
    a non-production axis between start and stop in step increments.

    Args:
        pro:
            A producer instance to slice.
        start:
            The start index of the slice. May be None.
        stop:
            The stop index of the slice. May be None.
        step:
            The step size between start and stop to slice with. May be None.
        axis:
            The non-production axis along which to slice.
    """

    for arr in pro:
        yield arraytools.slice_along_axis(arr, start, stop, step, axis=axis)


def mean(
    pro: Producer,
    axis: int = -1,
    ignore_nan=True,
    keepdims: bool = False,
) -> float | npt.NDArray:
    """Returns the mean of a producers values along axis.

    Args:
        pro:
            A producer instance.
        axis:
            The axis along which the mean will be computed.If this axis matches
            the production axis, the mean will be computed across all produced
            arrays.
        ignore_nan:
            Boolean indicating if produced nans should be ignored. Default is
            True.
        keepdims:
            Boolean indicating if number of output dimensions should match
            producers dimensionality. Default is False.

    Returns:
        A float or array of float values.
    """

    averager = np.nanmean if ignore_nan else np.mean

    ax = arraytools.normalize_axis(axis, pro.ndim)
    if pro.axis == ax:
        sums, cnts = 0, 0
        for arr in pro:
            cnts += arr.shape[axis]
            sums += arr.shape[axis] * averager(arr, axis=axis, keepdims=keepdims)

        return sums / cnts

    # non-production axis computes on each arr indptly.
    avgs = [averager(x, axis=ax, keepdims=True) for x in pro]
    result = np.concat(avgs, axis=pro.axis)
    if not keepdims:
        result = np.squeeze(result, ax)

    return result


def std(
    pro: Producer,
    axis: int = -1,
    ignore_nan=True,
    keepdims: bool = False,
) -> float | npt.NDArray:
    """Returns the standard deviation of a producer's values along axis.

    Args:
        pro:
            A producer instance.
        axis:
            The axis along which the standard deviation will be computed. If
            this axis matches the production axis, the std will be computed
            across all produced arrays.
        ignore_nan:
            Boolean indicating if produced nans should be ignored. Default is
            True.
        keepdims:
            Boolean indicating if number of output dimensions should match
            producers dimensionality. Default is False.

    Returns:
        A float or array of float values.
    """

    ax = arraytools.normalize_axis(axis, pro.ndim)
    averager = np.nanmean if ignore_nan else np.mean
    dev = np.nanstd if ignore_nan else np.std

    if ax == pro.axis:
        # compute squared expectation
        expected_squared = mean(pro, ax, ignore_nan, keepdims=keepdims) ** 2
        sum_squares, cnts = 0, 0
        for arr in pro:
            cnts += arr.shape[axis]
            sum_squares += arr.shape[axis] * averager(
                arr**2, axis=axis, keepdims=keepdims
            )

        # std = sqrt(E[x**2] - E[x]**2)
        return np.sqrt(sum_squares / cnts - expected_squared)

    # non-production axis computes on each arr indptly.
    stds = [dev(x, axis=ax, keepdims=True) for x in pro]
    result = np.concat(stds, axis=pro.axis)
    if not keepdims:
        result = np.squeeze(result, ax)

    return result


def standardize(
    pro: Producer,
    axis: int = -1,
    ignore_nan: bool = True,
) -> producer:
    """Standardizes the values in producer along axis.

    Examples:
    >>> z = np.random.random((3, 40))
    >>> y = (z - np.mean(z, axis=-1, keepdims=True)) / np.std(z, axis=-1,
    ...     keepdims=True)
    >>> apro = producer(z, chunksize=10, axis=-1)
    >>> standardized = standardize(apro, axis=-1)
    >>> x = standardized.to_array()
    >>> np.allclose(x, y)
    True

    Args:
        pro:
            A producer instance.
        axis:
            The axis along which the standardization will occur. If
            this axis matches the production axis, the mean and standard
            deviation from all produced arrays will be used.
        ignore_nan:
            Boolean indicating if produced nans should be ignored. Default is
            True.
        keepdims:
            Boolean indicating if number of output dimensions should match
            producers dimensionality. Default is False.

    Returns:
        A producer of standardized value.
    """

    means = mean(pro, axis, ignore_nan, keepdims=True)
    stds = std(pro, axis, ignore_nan, keepdims=True)
    func = partial(_standardize_gen, pro, means, stds, axis)

    return producer(func, pro.chunksize, pro.axis, shape=pro.shape)


def _standardize_gen(
    pro: Producer,
    means: npt.NDArray,
    stds: npt.NDArray,
    axis: int,
) -> Iterator[npt.NDArray]:
    """A generating helper function for producer standardization.

    Args:
        pro:
            A producer instance to standardize.
        means:
            The mean of the produced values along axis.
        stds:
            The standard deviations of the produced values along axis.
        axis:
            The axis along which standardization occurs. This should match the
            axis along which means and stds were computed.

    Yields:
        A standardized array on per array in pro.
    """

    if arraytools.normalize_axis(axis, pro.ndim) == pro.axis:
        for arr in pro:
            s = (arr - means) / stds
            yield s
    else:
        mean_pro = producer(means, chunksize=pro.chunksize, axis=pro.axis)
        std_pro = producer(stds, chunksize=pro.chunksize, axis=pro.axis)
        for arr, mu, dev in zip(pro, mean_pro, std_pro):
            yield (arr - mu) / dev
