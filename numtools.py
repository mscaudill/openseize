import numpy as np
from itertools import zip_longest
from scipy import fft, ifft

class IArrays:
    """This will be the iterable arrays class that IEDF is based on. It will
    have a data attr and a chunksize and an overidable iter method"""

    pass

def zero_pad(arr, pad, axis, **kwargs):
    """ """
    
    pads = [(0,0)] * arr.ndim
    pads[axis] = pad
    return np.pad(arr, pads, **kwargs)

def _optimal_nffts(arr):
    """ """

    return int(8 * 2 ** np.ceil(np.log2(len(arr))))
   
def _oa_applymode(segment, idx, total, win_len, mode):
    """ """

    modes = ('full', 'same', 'valid')
    if mode not in modes:
        msg = 'mode {} is not one of the valid modes {}'
        raise ValueError(msg.format(mode, modes))
    #remove zero pads
    nsamples = segment.shape[-1]
    if mode == 'full':
        result = segment
    if mode == 'same':
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

def _oa_arr(arr, win, axis, mode):
    """Performs overlap-add circular convolution of a window on a 2-D array.

    Args:
        arr (2-D array):            array of data to be convolved
        win (1-D array):            window array to convolve arr with
        axis (int):                 axis to apply convolution on
        mode FIXME
        
    Returns: generator of arrays
    """

    arr = arr.T if axis == 0 else arr
    nsamples = arr.shape[-1]
    #est. nffts, transform window, & set segment size
    nfft = _optimal_nffts(win)
    W = fft.fft(win, nfft)
    step = nfft - len(win) - 1
    nsegments = int(np.ceil(arr.shape[-1] / step))
    #overlap goes from step to nfft
    overlap = np.zeros((arr.shape[0], nfft - step))
    #perform overlap-add over each segment of size step
    starts = range(0, arr.shape[1], step)
    segments = zip_longest(starts, starts[1:], fillvalue=nsamples)
    for segment_num, (start, stop) in enumerate(segments):
        #extract and pad the segment upto nfft
        subarr = arr[:, start:stop]
        subarr = zero_pad(subarr, [0, nfft - step], axis=-1)
        #perform circular convolution
        y = fft.ifft(fft.fft(subarr, nfft) * W).real
        #split filtered segment and overlap
        y, new_overlap = np.split(y, [step], axis=-1)
        #add previous overlap to filtered segment & update overlap
        y[:, 0:nfft-step] += overlap
        overlap = new_overlap
        #last step may not have step pts left so slice result
        samples = min((stop-start) + len(win) -1, step)
        y = y[:, :samples]
        y = _oa_applymode(y, segment_num, nsegments, len(win), mode=mode) 
        yield y

def _oa_iarrays(iarrays, win, axis, mode):
    """ """

    pass

