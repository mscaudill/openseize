import numpy as np

def pad_along_axis(arr, pad, axis, **kwargs):
    """Wrapper for numpy pad allowing before and after padding along
    a single axis.

    Args:
        arr (ndarray):              ndarray to pad
        pad (int or array-like):    number of pads to apply before the 0th
                                    and after the last index of array along 
                                    axis. If int, pad number of pads will be 
                                    added to both
        axis (int):                 axis of arr along which to apply pad
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


