import copy
import numpy as np
from itertools import zip_longest
from scipy import fft, ifft

def zero_pad(arr, pad, axis, **kwargs):
    """ """
    
    pads = [(0,0)] * arr.ndim
    pads[axis] = pad
    return np.pad(arr, pads, **kwargs)


def oaconvolve(signal, win, axis, mode):
    """ """

    pass

def _optimal_nffts(arr):
    """ """

    return int(8 * 2 ** np.ceil(np.log2(len(arr))))
   
def _oa_applymode(segment, idx, total, win_len, mode):
    """ """

    modes = ('full', 'same', 'valid')
    if mode not in modes:
        msg = 'mode {} is not one of the valid modes {}'
        raise ValueError(msg.format(mode, modes))
    nsamples = segment.shape[-1]
    if mode == 'full':
        result = segment
    if mode == 'same':
        #remove zero pads
        result = segment
        if idx == 0:
            result = segment[:, (win_len-1)//2:]
        elif idx + 1 == total:
            result = segment[:, :nsamples - (win_len-1)//2]
    if mode == 'valid':
        result = segment
        #remove zero pads and samples that were computed using them
        if idx == 0:
            result = segment[:, win_len-1:]
        elif idx + 1 == total:
            result = segment[:, :nsamples - win_len + 1]
    return result

def _oa_array(arr, win, axis, mode):
    """A generator that performs overlap-add circular convolution of an
    array with a 1-dimensional window.

    Args:
        arr (ndarray):              a n-dim numpy array to convolve
        win (1-D array):            a 1-D window to convolve across arr
        axis (int):                 axis of arr along which window should be
                                    convolved
        mode (str):                 one of 'full', 'same', 'valid'.
                                    identical to numpy convovle mode
    """

    nsamples = arr.shape[axis]
    #estimate optimal nfft and transform window
    nfft = _optimal_nffts(win)
    W = fft.fft(win, nfft)
    #set the step size and compute num of segments of step size
    step = nfft - len(win) -1
    nsegments = int(np.ceil(arr.shape[axis] / step))
    #initialize the overlap with shape (nfft - step) along axis
    overlap_shape = list(arr.shape)
    overlap_shape[axis] = nfft - step
    overlap = np.zeros(overlap_shape)
    #create segments of steps size for overlap-add
    starts = range(0, arr.shape[axis], step)
    segments = zip_longest(starts, starts[1:], fillvalue=nsamples)
    slices = [slice(None)] * arr.ndim
    for segment_num, (start, stop) in enumerate(segments):
        #extract and pad the step size subarr and pad upto nfft
        slices[axis] = slice(start, stop)
        subarr = zero_pad(arr[tuple(slices)], [0, nfft - step], axis=axis)
        #perform circular convolution
        y = fft.ifft(fft.fft(subarr, nfft, axis=axis) * W, axis=axis).real
        #split filtered segment and overlap
        y, new_overlap = np.split(y, [step], axis=axis)
        #add previous overlap to convolved (reuse slices list)
        slices[axis] = slice(0, nfft-step)
        y[tuple(slices)] += overlap
        #update overlap for next segment
        overlap = new_overlap
        #last step may not have step pts left so slice result (reuse slices)
        samples = min((stop-start) + len(win) -1, step)
        slices[axis] = slice(None, samples)
        y = y[tuple(slices)]
        #apply the boundary mode (same defn as numpy & scipy)
        y = _oa_applymode(y, segment_num, nsegments, len(win), mode=mode)
        yield y



        
