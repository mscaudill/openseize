import copy
import numpy as np
from itertools import zip_longest
from scipy import fft, ifft

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
        slices[axis] = slice(None, samples)
        y = y[tuple(slices)]
        #apply the boundary mode to first and last segments
        if segment_num == 0 or segment_num == nsegments-1:
            y = _oa_mode(y, segment_num, len(win), axis, mode)
        yield y

        
