import numpy as np
import itertools
from scipy import fft, ifft
import scipy.signal as sps

from openseize.types.producer import Producer, producer 

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

def optimal_nffts(arr):
    """Estimates the optimal number of FFT points for an arr."""

    return int(8 * 2 ** np.ceil(np.log2(len(arr))))
   
def _oa_mode(segment, idx, win_len, axis, mode):
    """Applies the numpy/scipy mode to the first and last segement of the
    oaconvolve generator.

    Args:
        segment (arr):          array of values yielded by oaconvolve
        idx (int):              index number of the segment
        total (int):            total number of segments 
        win_len (int):          len of convolving window
        axis (int):             axis along which convolution was applied
        mode (str):             one of 'full', 'same', 'valid'.
                                identical to numpy convovle mode
    
    Returns: segment with boundary mode applied
    """

    modes = ('full', 'same', 'valid')
    if mode not in modes:
        msg = 'mode {} is not one of the valid modes {}'
        raise ValueError(msg.format(mode, modes))

    ns = segment.shape[axis]
    #dict of slices to apply to axis for 0th & last segment for each mode
    cuts = {'full': [slice(None), slice(None)],
            'same': [slice((win_len - 1) // 2, None), 
                     slice(None, ns - ((win_len - 1) // 2))],
            'valid': [slice(win_len - 1, None), 
                      slice(None, ns - win_len + 1)]}
    #apply slices
    slices = [slice(None)] * segment.ndim
    slices[axis] = cuts[mode][1] if idx > 0 else cuts[mode][0]
    return segment[tuple(slices)]

def oaconvolve(iterable, win, axis, mode):
    """A generator that performs overlap-add circular convolution of a
    an array or producer of arrays with a 1-dimensional window.

    Args:
        iterable:                   an ndarray or a producer of ndarrays
        win (1-D array):            a 1-D window to convolve across arr
        axis (int):                 axis of arr or each producer array
                                    along which window should be convolved
        mode (str):                 one of 'full', 'same', 'valid'.
                                    identical to numpy convolve mode

    Numpy & Scipy implement oaconvolve but require that the input and  
    output be storable in RAM. This implementation makes no such constraint.
    The input iterable can be an in-memory array, a numpy memmap, a 
    generator of arrays or an iterable of arrays (see producer). 

    Returns: a generator of convolved arrays. The length of each array along
    axis will be (optimal_nffts - len(win) - 1).
    """

    #create a producer, chunksize will be changed later
    pro = producer(iterable, chunksize=1, axis=axis)
    #estimate optimal nfft and transform window
    nfft = optimal_nffts(win)
    W = fft.fft(win, nfft)
    #set the step size and compute num of segments of step size
    step = nfft - len(win) -1
    nsegments = int(np.ceil(pro.shape[axis] / step))
    #initialize the overlap with shape (nfft - step) along axis
    overlap_shape = list(pro.shape)
    overlap_shape[axis] = nfft - step
    overlap = np.zeros(overlap_shape)
    #set producer chunksize to step & perform overlap add
    pro.chunksize = step
    for segment_num, subarr in enumerate(pro):
        #pad the subarr upto nfft
        x = pad_along_axis(subarr, [0, nfft - step], axis=axis)
        #perform circular convolution
        y = fft.ifft(fft.fft(x, nfft, axis=axis) * W, axis=axis).real
        #split filtered segment and overlap
        y, new_overlap = np.split(y, [step], axis=axis)
        slices = [slice(None)] * subarr.ndim
        #add previous overlap to convolved
        slices[axis] = slice(0, nfft - step)
        y[tuple(slices)] += overlap
        #update overlap for next segment
        overlap = new_overlap
        #last segment may not have step pts, so slice upto 'full' overlap
        samples = min((subarr.shape[axis]) + len(win) -1, step)
        y = slice_along_axis(y, 0, samples, axis=axis)
        #apply the boundary mode to first and last segments
        if segment_num == 0 or segment_num == nsegments-1:
            y = _oa_mode(y, segment_num, len(win), axis, mode)
        yield y

def batch_sosfilt(sos, iterable, chunksize, axis, zi=None):
    """Batch applies a second-order-section fmt filter to a data iterable.

    Args:
        sos (array):             a nsectios x 6 array of numerator &
                                 denominator coeffs from an iir filter
        iterable:                an ndarray or producer of ndarrays (e.g
                                 generator or Producer instance)
        chunksize (int):         amount of data along axis to filter per
                                 batch
        axis (int):              axis along which to apply the filter in
                                 chunksize batches

    Returns: a generator of filtered values of len chunksize along axis
    """

    #create a producer yielding chunksize arrays
    pro = producer(iterable, chunksize=chunksize, axis=axis)
    #set initial conditions for filter (see scipy sosfilt; zi)
    shape = list(pro.shape)
    shape[axis] = 2
    z = np.zeros((sos.shape[0], *shape)) if zi is None else zi
    #compute filter values & store current initial conditions 
    for idx, subarr in enumerate(pro):
        y, z = sps.sosfilt(sos, subarr, axis=axis, zi=z)
        yield y

def batch_sosfiltfilt(sos, iterable, chunksize, axis):
    """Batch applies a forward-backward filter in sos format to an iterable
    of numpy arrays.

    Args:
        sos (array):             a nsectios x 6 array of numerator &
                                 denominator coeffs from an iir filter
        iterable:                an ndarray or producer of ndarrays (e.g
                                 generator or Producer instance)
        chunksize (int):         amount of data along axis to filter per
                                 batch
        axis (int):              axis along which to apply the filter in
                                 chunksize batches

    Returns: a generator of forward-backward filtered values of len 
             chunksize along axis
    """

    #create a producer yielding chunksize arrays
    pro = producer(iterable, chunksize=chunksize, axis=axis)
    #get initial value from producer
    subarr = next(iter(pro))
    x0 = slice_along_axis(subarr, 0, 1, axis=axis) 
    # build initial condition
    zi = sps.sosfilt_zi(sos)
    s = [1] * len(pro.shape)
    s[axis] = 2
    zi = np.reshape(zi, (sos.shape[0], *s))
    #create a producer of forward filter values
    forward = batch_sosfilt(sos, pro, chunksize, axis, zi=zi * x0)
    #get the last value of the forward filtered values
    forward_filt, tmp = itertools.tee(forward)
    y0 = next(reversed(producer(tmp, chunksize=1, axis=axis)))
    #compute reverse filter values
    rev_gen = reversed(producer(forward_filt, chunksize, axis))
    rev_filt = batch_sosfilt(sos, rev_gen, chunksize, axis, zi=zi * y0)
    #reverse the rev_filt values to get the filtfilt values
    return reversed(producer(rev_filt, chunksize, axis))

def batch_iirfilter(coeffs, iterable, chunksize, axis, compensate):
    """Performs batch filtering of an array iterable.

    Args:
        coeffs (array-like (sos) or tuple (ba), (zpk))

    """

    # validate the coeffs are of fmt sos, ba or zpk -- need validator
    # later add lfilter and lfiltfilt but now raise not Implemented error

    pass
