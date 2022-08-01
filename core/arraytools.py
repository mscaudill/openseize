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

def expand_along_axis(arr, l, value=0, axis=-1):
    """Inserts l-1 copies of value between samples in an array along axis.

    Args:
        arr (ndarray):              an ndarray to expand
        l (int):                    the expansion factor. l-1 replicates of
                                    value will be inserted betwee samples
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

    Returns: 1-D array of False values everwhere except indices

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


