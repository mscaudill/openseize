import numpy as np
import itertools
from functools import partial
from scipy import fft, ifft
import scipy.signal as sps

from openseize.core.producer import Producer, producer, as_producer 
from openseize.core.arraytools import pad_along_axis, slice_along_axis

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


@as_producer
def oaconvolve(pro, win, axis, mode):
    """A generator that performs overlap-add circular convolution of a
    an array or producer of arrays with a 1-dimensional window.

    Args:
        pro:                        producer of ndarray(s)
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

    #fetch the producers initial chunksize
    isize = pro.chunksize

    #estimate optimal nfft and transform window
    nfft = optimal_nffts(win)

    # faster if we pad upto nfft 5-11-2022
    w = np.pad(win, [0, nfft - len(win)]) # new to pad window

    #W = fft.fft(win, nfft) original
    W = fft.fft(w, nfft) #5-11-2022

    #set the step size and compute num of segments of step size
    step = nfft - len(win) + 1
    nsegments = int(np.ceil(pro.shape[axis] / step))

    #initialize the overlap with shape (nfft - step) along axis
    overlap_shape = list(pro.shape)
    overlap_shape[axis] = nfft - step
    overlap = np.zeros(overlap_shape)

    #set producer chunksize to step & perform overlap add
    pro.chunksize = step
    for segment_num, subarr in enumerate(pro):
        print(segment_num, subarr.shape)
        islast = (segment_num == nsegments - 1)
        #pad the subarr upto nfft
        #x = pad_along_axis(subarr, [0, nfft - step], axis=axis) #original
        x = pad_along_axis(subarr, [0, len(win) - 1], axis=axis)
        if islast:
            print('padding last')
            pad_amt = step - x.shape[axis] + len(win) -1
            print(pad_amt)
            x = pad_along_axis(x, [0, pad_amt], axis=axis)
        #perform circular convolution
        y = fft.ifft(fft.fft(x, nfft, axis=axis) * W, axis=axis).real
        #split filtered segment and overlap
        if not islast:
            y, new_overlap = np.split(y, [step], axis=axis)
        slices = [slice(None)] * subarr.ndim
        #add previous overlap to convolved
        slices[axis] = slice(0, nfft - step)
        y[tuple(slices)] += overlap
        #update overlap for next segment
        overlap = new_overlap
        #last segment may not have step pts, so slice upto 'full' overlap
        
        """
        samples = min((subarr.shape[axis]) + len(win) -1, step)
        y = slice_along_axis(y, 0, samples, axis=axis)
        """
        
        print(segment_num, y.shape)
        """
        #apply the boundary mode to first and last segments
        if segment_num == 0 or segment_num == nsegments-1:
            y = _oa_mode(y, segment_num, len(win), axis, mode)
        """
        yield y

    #on iteration completion reset producer chunksize
    pro.chunksize = isize

# FIXME Improve docs around zi please see iir apply method
@as_producer  
def sosfilt(pro, sos, chunksize, axis, zi=None):
    """Batch applies a second-order-section fmt filter to a producer.

    Args:
        pro:                     producer of ndarrays
        sos (array):             a nsectios x 6 array of numerator &
                                 denominator coeffs from an iir filter
        chunksize (int):         amount of data along axis to filter per
                                 batch
        axis (int):              axis along which to apply the filter in
                                 chunksize batches
        zi (ndarray):            initial condition data (Default None ->
                                 zeros as intial condition)

    Returns: a generator of filtered values of len chunksize along axis
    """

    #set initial conditions for filter (see scipy sosfilt; zi)
    shape = list(pro.shape)
    shape[axis] = 2
    z = np.zeros((sos.shape[0], *shape)) if zi is None else zi
    #compute filter values & store current initial conditions 
    for idx, subarr in enumerate(pro):
        y, z = sps.sosfilt(sos, subarr, axis=axis, zi=z)
        yield y


@as_producer
def sosfiltfilt(pro, sos, chunksize, axis):
    """Batch applies a forward-backward filter in sos format to a producer
    of numpy arrays.

    Args:
        pro:                     producer of ndarrays
        sos (array):             a nsectios x 6 array of numerator &
                                 denominator coeffs from an iir filter
        chunksize (int):         amount of data along axis to filter per
                                 batch
        axis (int):              axis along which to apply the filter in
                                 chunksize batches

    Returns: a generator of forward-backward filtered values of len 
             chunksize along axis
    """

    #get initial value from producer
    subarr = next(iter(pro))
    x0 = slice_along_axis(subarr, 0, 1, axis=axis) 
    
    # build initial condition
    zi = sps.sosfilt_zi(sos) #nsections x 2
    s = [1] * len(pro.shape)
    s[axis] = 2
    zi = np.reshape(zi, (sos.shape[0], *s)) #nsections,1,2
    
    #create a producer of forward filter values
    forward = sosfilt(pro, sos, chunksize, axis, zi=zi*x0)
    
    #filter backwards each forward produced arr
    for idx, arr in enumerate(forward):
        flipped = np.flip(arr, axis=axis)
        #get the last value as initial condition for backward pass
        y0 = slice_along_axis(flipped, 0, 1, axis=axis)
        #filter in reverse, reflip and yield
        revfilt, z = sps.sosfilt(sos, flipped, axis=axis, zi=zi*y0)
        yield np.flip(revfilt, axis=axis)

# FIXME
# IMPLEMENT filtfilt for transfer function format filters
# IMPLEMENT lfilter for transfer function format filters

def welch(pro, fs, nperseg, window, axis, csize=100, **kwargs):
    """Iteratively estimates the power spectral density of data from
    a producer using Welch's method.

    Args:
        pro: producer instance
                a producer of ndarrays

        fs: float
                sampling rate of produced data

        nperseg: int
                number of points used to compute FFT on each segment. Should
                be < pro.chunksize and ideally a power of 2

        window: str or tuple or array_like
                a scipy window string or 1-D array of window heights. If 
                array_like, its length must match nperseg

        axis: int
                axis of producer along which spectrum is computed

        csize: int
                factor to multipy nperseg by that will be used as the
                chunksize of the producer during welch computation. Default
                is 100. The producer's chunksize will be set back to initial
                chunksize after PSD is computed.

        kwargs: passed to scipy.signal.welch

    Returns:
        f: ndarray
                array of FFT sample frequencies

        pxx: ndarray
                power spectral density or power spectrum of producer

    See also:
        scipy.signal.welch

    References:
    1.  P. Welch, “The use of the fast Fourier transform for the 
        estimation of power spectra: A method based on time averaging over
        short, modified periodograms”, IEEE Trans. Audio Electroacoust. 
        vol. 15, pp. 70-73, 1967.

    Notes:
        Masked producers may produce arrays with few if any values. Scipy 
        sets the nperseg parameter to the min. of the input data & nperseg.
        This behavior changes the FFT frequency resolution depending on the
        amount of data in each produced array. We therefore drop segments
        from the producer whose size along axis is less than nperseg.
    """

    # chunksize will need to be a multiple of nperseg to avoid gaps
    # need to get last overlap window from previous array to have 
    # continuous overlapping windows

    overlap = kwargs.pop('noverlap', nperseg // 2)
    isize = pro.chunksize
    pro.chunksize = csize * nperseg

    cnt, current_sum = 0, 0
    previous = np.array([])
    for arr in pro:
        
        if previous.size > 0:
            x = np.concatenate((previous, arr), axis=axis)
        else:
            x = arr

        if x.shape[axis] >= nperseg:
            f, _, pxx = sps.spectral._spectral_helper(x, x, fs=fs, 
                            window=window, nperseg=nperseg, axis=axis, 
                            mode='psd', **kwargs)
            #update number of windows and add to summed psd
            cnt += pxx.shape[-1]
            current_sum += np.sum(pxx, axis=-1)
            previous = slice_along_axis(x, x.shape[axis]-overlap, axis=axis)

    pro.chunksize = isize
    return f, current_sum / cnt

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from openseize.io.readers import EDF
    from openseize.types.producer import producer

    import time

    PATH = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    edf = EDF(PATH)
    pro = producer(edf, chunksize=10e6, axis=-1)

    t0 = time.perf_counter()
    f, pxx = welch(pro, 5000, nperseg=16384, window='hann', axis=-1,
            csize=1000)
    print('ops welch in {} s'.format(time.perf_counter() - t0))

    """
    x = np.concatenate([arr for arr in pro], axis=-1)
    t0 = time.perf_counter()
    sp_f, sp_pxx = sps.welch(x, fs=5000, nperseg=16384, window='hann', axis=-1)
    print('sp welch in {} s'.format(time.perf_counter() - t0))
    """

    plt.plot(f, pxx[0], label='os result')
    """
    plt.plot(f, sp_pxx[0], label='sp result')
    plt.plot(f, (pxx[0] - sp_pxx[0]), label='residual')
    plt.legend()
    plt.show()
    """
