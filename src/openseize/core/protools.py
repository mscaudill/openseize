"""A collection of tools to manipulate the size, shape or values produced by
a producer including:

    pad:
        A function to pre and post pad a producer along a single axis.
    expand_dims:
        A function that expands a producers shape by axis insertion.
    multiply_along_axis:
        A function that multiplies produced values by a 1-D numpy array along
        a single axis.
    
Note: To support concurrency all functions in this module are available at the
module level. Functions not intended to be called externally are marked as 
protected with a single underscore.
"""

from typing import Optional, Sequence, Tuple, Union
from functools import partial

import numpy as np
import numpy.typing as npt
from openseize import producer
from openseize.core import arraytools
from openseize.core import Producer


def pad(pro: Producer,
        pad: Union[int, Sequence[int, int]],
        axis: int,
        value: Optional[float] = 0,
) -> Producer:
    """Pads the edges of a producer along axis with a constant value.

    Args:
        pro:
            A producer of ndarrays whose edges along axis are to be padded.
        pad:
            The number of pads to apply before the 0th element and after the
            last element along axis. If int, pad number of values will be
            prepended and appended to axis.
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

    pad = (pad, pad) if isinstance(pad, int) else tuple(pad)
   
    # dispatch to generating function based on wheter pad is along pro.axis
    if axis == pro.axis:
        genfunc = _production_axis_padder
    else:
        genfunc = _other_axis_padder
   
    # build a partial generating function and compute the return pros shape
    func = partial(genfunc, pro, pad, axis, value)
    new_shape = list(pro.shape)
    new_shape[axis] = pro.shape[axis] + sum(pads)
    
    return producer(func, pro.chunsize, pro.axis, shape=new_shape)
    

def _production_axis_padder(pro, pad, axis, value):
    """A generating func. that pads a producer along its axis with value.

    Padding a producer along its production axis only changes the first and last
    produced arrays. For argument definitions see pad.
    """

    left_shape, right_shape = list(pro.shape), list(pro.shape)
    left_shape[axis] = pad[0]
    right_shape[axis] = pad[1]

    # create the arrays to pad left and right along axis
    left, right = [value * np.ones(s) for s in (left_shape, right_shape)]

    yield left

    for arr in pro:
        yield arr

    yield right


def _other_axis_padder(pro, pad, axis, value):
    """A generating func. that pads a producer along any non-production axis.

    Padding a producer along an axis that does not match the producers axis
    changes all produced arrays.
    """
    
    for arr in pro:
        yield = arraytools.pad_along_axis(arr, pad, axis, constant_values=value)


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
        -1
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
        produced array dims and track where the producing axis ends up.
    """

    # normalize the axis to insert and the producer's axis
    axes = (axis,) if isinstance(axis, int) else axis
    pro_axis = normalize_axis(pro.axis, len(pro.shape))

    # calculate out ndims, initialize new shape and normalize inserts
    new_ndim = len(pro.shape) + len(axes)
    new_shape = np.ones(new_ndim, dtype=int)
    inserts = [normalize_axis(ax, new_ndim) for ax in axes]

    # find indices of new_shape where we will insert producer's shape
    complements = sorted(set(range(new_ndim)).difference(inserts))

    # set the new axis and insert producer's shape into new shape
    new_axis = complements[pro_axis]

    for idx, comp in enumerate(complements):

        new_shape[comp] = pro.shape[idx]

    func = partial(_expand_gen, pro, axes)
    return producer(func, pro.chunksize, new_axis, new_shape)


def _expand_gen(pro: Producer, axes: Tuple[int, ...]):
        """A generating function that expands the dims of each produced array
        in a producer.

        This helper function is a generating function (not a producer) and is not
        intended to be called externally.

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


def multiply_along_axis(pro: Producer, arr: npt.NDArray, axis: int,
) -> Producer:
    """Multiplies each produced array of a producer by a 1-D array 
    along a single axis.

    Args:
        pro:
            A producer of ndarrays
        arr:
            A 1-D array whose length must match producers shape along the
            supplied axis.
        axis:
            The axis along which to multiply.

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
    # ensure the arr shape matches the producers shape along axis
    if len(arr) != pro.shape[axis]:
        msg = 'operands could not be broadcast together with shapes {} {}'
        raise ValueError(msg.format(pro.shape, arr.shape))

    # reshape the input array to be broadcastable with produced arrays
    ndims = len(pro.shape)
    shape = np.ones(ndims, dtype=int)
    shape[axis] = len(arr)
    y = arr.reshape(shape)

    func = partial(_multiply_gen, pro, y)
    return producer(func, chunksize=pro.chunksize, axis=pro.axis,
                    shape=pro.shape)


def _multiply_gen(pro: Producer, arr: npt.NDArray):
    """A generating helper function that multiplies produced arrays by an 
    ndarray.

    This helper function is a generating function (not a producer) and is not
    intended to be called externally.

    Args:
        pro:
            A producer of ndarrays.
        arr:
            An ndarray of the same dims as each produced array.

    Yields:
        The element-wise product of each produced array with arr.
    """

    for x in pro:
        yield x * arr

