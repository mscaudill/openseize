import numpy as np


def pad_along_axis(arr, pad, axis=-1, **kwargs):
    """Wrapper for numpy pad allowing before and after padding along
    a single axis.

    Args:
        arr (ndarray):              ndarray to pad
        pad (int or array-like):    number of pads to apply before the 0th
                                    and after the last index of array along 
                                    axis. If int, pad number of pads will be 
                                    added to both
        axis (int):                 axis of arr along which to apply pad.
                                    Default pads along last axis.
        **kwargs:                   any valid kwarg for np.pad
    """
    
    #convert int pad to seq. of pads & place along axis of pads
    pad = [pad, pad] if isinstance(pad, int) else pad
    pads = [(0,0)] * arr.ndim
    pads[axis] = pad
    return np.pad(arr, pads, **kwargs)


def slice_along_axis(arr, start=None, stop=None, step=None, axis=-1):
    """Returns slice of arr along axis from start to stop in 'step' steps.

    (see scipy._arraytools.axis_slice)

    Args:
        arr (ndarray):              an ndarray to slice
        start, stop, step (int):    passed to slice instance
        axis (int):                 axis of array to slice along

    Returns: sliced ndarray
    """

    slicer = [slice(None)] * arr.ndim
    slicer[axis] = slice(start, stop, step)
    return arr[tuple(slicer)]


def split_along_axis(arr, index, axis=-1):
    """Returns two arrays by splitting arr along axis at index.

    Args:
        arr: ndarray
            An ndarray to split.
        index: int
            The index to split on. It is excluded from first split array 
            and included in second split arr.
        axis: int
            Axis along which to split the array.

    Returns: Two ndarray split at index.
    
    Note: This method uses slicing instead of numpy split as its performance
    is better.
    """

    first = slice_along_axis(arr, start=0, stop=index, axis=axis)
    second = slice_along_axis(arr, start=index, stop=None, axis=axis)

    return first, second


def expand_along_axis(arr, l, value=0, axis=-1):
    """Inserts l-1 copies of value between samples in an array along axis.

    Args:
        arr (ndarray):              an ndarray to expand
        l (int):                    the expansion factor. l-1 replicates of
                                    value will be inserted between samples
        value (float):              value to insert between samples
        axis (int):                 axis along which to expand

    Returns:
        An ndarray whose size matches arr except along expansion axis. The
        new size along this axis will be l * arr.shape[axis].
    """

    # move insertion axis to last axis
    x = np.swapaxes(arr, axis, -1)
    init_shape = x.shape

    # perform insertion into flattened x
    x = x.reshape(-1, 1)
    inserts = value * np.ones((x.shape[0], l-1))
    x = np.concatenate((x, inserts), axis=-1)
    x = x.flatten()

    # reshape back to initial shape but with added samples along -1 axis 
    x = x.reshape(*init_shape[:-1], -1)

    # move the -1 axis back to its original position
    x = np.swapaxes(x, -1, axis)
    return x


def multiply_along_axis(x, y, axis=-1):
    """Multiplies an ndarray by a 1-D array along axis.
    
    Args:
        x: ndarray
            Array to multiply along axis.
        y: 1-D array
            Values to multiply arr by along axis.
        axis: int
            axis along which to multiply
            
    Returns: ndarray of shape x.
    """

    shape = np.ones(x.ndim, int)
    shape[axis] = len(y)
    return x * y.reshape(shape)

def filter1D(size, indices):
    """Returns a 1D array of False values except at indices where values are
    True.

    Args:
        size (int):         length of the returned array
        indices:            an object or list objects that can be used to
                            slice or index a 1D array. This includes:
                            slice obj, list, list of slices or list of lists

    Usage:
    >>> x = filter1D(60, [slice(0,8), slice(11,13)])
    >>> y = filter1D(60, slice(0,4))
    >>> p = filter1D(60, [1,5,8])
    >>> q = filter1D(60, [[2,3,4], [7,11, 21, 22]])

    Returns: 1-D array of False values everywhere except indices

    # TODO: In future this will support ndarray filtering
    """

    locs = np.atleast_1d(np.array(indices, dtype='object'))
    result = np.full(int(size), fill_value=False, dtype=bool)
    for idxs in locs:
        result[idxs] = True
    return result


def nearest1D(x, x0, axis=-1):
    """Returns the index in a 1-D array whose value has the smallest 
    distance to x0

    Args:
        x: 1-D array
            A sequence of values to search for minimim distance to x0.
        x0: float
            The value whose distance from each value in x is measured.
    
    Returns: 
        The index of x whose distance to x0 is smallest.
    """

    index = np.argmin(np.abs(x - x0))
    return index


def zero_extend(arr, n, axis=-1):
    """Pads an array on both ends with zeros along axis.

    Args:
        arr: ndarray
            Array to be extended with zeros.
        n: int
            The number of zeros to extend arr at both ends along axis.
        axis: int
            The axis along which to extend arr.

    Returns: an ndarray with n repeats of zeros at ends of axis.

    Typical Usage Example:

    >>> from openseize.core.arraytools import zero_extend
    >>> x = np.array([[1,2,3], [4,5,6]])
    >>> zero_extend(x, n=2, axis=-1)
    array([[0, 0, 1, 2, 3, 0, 0], [0, 0, 4, 5, 6, 0, 0]])
    """

    return pad_along_axis(arr, n, axis=axis)


def edge_extend(arr, n, axis=-1):
    """Pads an array with the edge values along axis.

    Args:
        arr: ndarray
            Array to be extended with edge values.
        n: int
            The number of times to repeat edge values for each end of axis.
        axis: int
            The axis along which to extend arr.

    Returns: an ndarray with n-repeats of edge values at the ends of axis.

    Typical Usage Example:

    >>> from openseize.core.arraytools import edge_extend
    >>> x = np.array([[1,2,3], [4,5,6]])
    >>> edge_extend(x, n=2, axis=-1)
    array([[1, 1, 1, 2, 3, 3, 3], [4, 4, 4, 5, 6, 6, 6]])
    """

    left = slice_along_axis(arr, start=0, stop=1, axis=axis)
    right = slice_along_axis(arr, start=-1, stop=None, axis=axis)
    left, right = [np.repeat(x, n, axis=axis) for x in (left, right)]
    
    return np.concatenate((left, arr, right), axis=axis)


def even_extend(arr, n, axis=-1):
    """Pads an array with a mirror image of edge values along axis.

    Args:
        arr: ndarray
            Array to be extended with even-symmetry edge values.
        n: int
            The number of additional edge values for each end of axis.
        axis: int
            The axis along which to extend arr.

    Returns: an ndarray with n even-symmetric edge values concatenated to
    each end of axis.

    Typical Usage Example:

    >>> from openseize.core.arraytools import even_ext
    >>> x = np.array([[1,2,3], [4,5,6]])
    >>> even_extend(x, n=2, axis=-1)
    array([[3, 2, 1, 2, 3, 2, 1], [6, 5, 4, 5, 6, 5, 4]])
    """

    # for consistency match scipy error msg
    if n > arr.shape[axis] - 1:
        msg = ('The extension length n ({}) is too big. It must not '
               'exceed x.shape[axis] - 1, which is {}.')
        raise ValueError(msg.format(n, arr.shape[axis] - 1))

    left = slice_along_axis(arr, start=n, stop=0, step=-1, axis=axis)
    right = slice_along_axis(arr, start=-2, stop=-(n+2), step=-1, axis=axis)

    return np.concatenate((left, arr, right), axis=axis)


def odd_extend(arr, n, axis=-1):
    """Pads an array with an odd extension of edge values along axis.

    Args:
        arr: ndarray
            Array to be extended with odd-symmetry edge values.
        n: int
            The number of additional edge values for each end of axis.
        axis: int
            The axis along which to extend arr.

    Returns: an ndarray with n odd-symmetric edge values concatenated to
    each end of axis.

    Typical Usage Example:

    >>> from openseize.core.arraytools import even_ext
    >>> x = np.array([[1, 2, 3, 4 , 5], [0, 2, -1, 3, 1]])
    >>> odd_extend(x, n=2, axis=-1)
    array([[-1, 0, 1, 2, 3, 4, 5, 6, 7], [1, -2, 0, 2, -1, 3, 1, -1, 3]])
    """

    # for consistency match scipy error msg
    if n > arr.shape[axis] - 1:
        msg = ('The extension length n ({}) is too big. It must not '
               'exceed x.shape[axis] - 1, which is {}.')
        raise ValueError(msg.format(n, arr.shape[axis] - 1))

    leftmost = slice_along_axis(arr, start=0, stop=1, axis=axis)
    rightmost = slice_along_axis(arr, start=-1, axis=axis)
    
    # get the boundary points to rotate about leftmost & rightmost
    left = slice_along_axis(arr, start=n, stop=0, step=-1, axis=axis)
    right = slice_along_axis(arr, start=-2, stop=-(n+2), step=-1, axis=axis)

    # 180-deg rotation is a mirror followed by flip about left/rightmost
    left_ext = (leftmost - left) + leftmost
    right_ext = (rightmost - right) + rightmost

    return np.concatenate((left_ext, arr, right_ext), axis=axis)

    


if __name__ == '__main__':

    from scipy.signal._arraytools import even_ext, odd_ext, const_ext

    
    x = np.array([[1,2,3, 4, 5], [0, 2, -1, 3, 1]])
    y = odd_extend(x, n=2, axis=-1)
    
    """ 
    rng = np.random.default_rng(33)
    x = rng.random((4, 12, 7, 100))

    y = edge_extend(x, n=17, axis=1)
    y_sp = const_ext(x, n=17, axis=1)
    """
  

    """
    y = even_extend(x, n=9, axis=-1)
    y_sp = even_ext(x, n=9, axis=-1)
    """


    """
    y = odd_extend(x, n=5, axis=2)
    y_sp = odd_ext(x, n=5, axis=2)
    """

    #print(np.allclose(y, y_sp))


