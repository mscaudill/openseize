import numpy as np
import itertools
from functools import partial
from numpy import fft
import scipy.signal as sps

from openseize.core.producer import Producer, producer, as_producer 
from openseize.core.arraytools import pad_along_axis, slice_along_axis
from openseize.filtering.fir import Kaiser

def optimal_nffts(arr):
    """Estimates the optimal number of FFT points for an arr."""

    return int(8 * 2 ** np.ceil(np.log2(len(arr))))

def convolve_slicer(arr, shape1, shape2, mode, axis):
    """Applies a boundary mode to slice a convolved array along axis.

    Args:
        arr: ndarray
            A convolved ndimensional array.
        shape1: tuple
            Shape of the first input used to construct the convolved arr.
        shape2: tuple
            Shape of the second input used to construct the convolved arr.
        mode: str one of 'full', 'same', or 'valid'
            A string defining the boundary handling mode. These modes
            are the same as numpy's np.convolve mode argument.
        axis: int
            The convolution axis in arr.

    Returns:
        An ndarray with the convolve mode applied.
    """

    m, n = shape1[axis], shape2[axis]
    p, q = max(m,n), min(m,n)

    if mode == 'full':

        # full mode length is m + n - 1
        return arr
    
    if mode == 'same':
        
        # same mode length is max(m,  n) centered along axis of arr
        start = (q - 1) // 2
        stop = start + p
        return slice_along_axis(arr, start, stop, axis=axis)

    elif mode == 'valid':
       
        # valid mode length is m + n - 1 - 2 * (q-1) centered along axis
        start= q - 1 
        stop = (n + m - 1) - (q - 1)
        return slice_along_axis(arr, start, stop, axis=axis)

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

    #FIXME refactor to reuse your convolve slicer

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
    axis will be (optimal_nffts(win) - len(win) - 1). See optimal_nffts
    func.
    """

    #fetch the producers initial chunksize
    isize = pro.chunksize

    #estimate optimal nfft and transform window
    nfft = optimal_nffts(win)
    M = len(win)
    H = fft.fft(win, nfft)

    #set the step size 'L' and compute num of segments
    L = nfft - M + 1
    nsegments = int(np.ceil(pro.shape[axis] / L))

    #initialize the overlap with shape  along axis
    overlap_shape = list(pro.shape)
    overlap_shape[axis] = M - 1
    overlap = np.zeros(overlap_shape)

    #set producer chunksize to L & perform overlap add
    pro.chunksize = L
    for segment_num, subarr in enumerate(pro):
        # pad the length L segment with M-1 zeros for overlap
        x = pad_along_axis(subarr, [0, M - 1], axis=axis)
        
        #perform circular convolution
        y = fft.ifft(fft.fft(x, nfft, axis=axis) * H, axis=axis).real
        
        #split filtered segment and overlap
        if segment_num < nsegments - 1:
            y, new_overlap = np.split(y, [L], axis=axis)
        else:
            # last sample for last segment is data + window size
            last = subarr.shape[axis] + M - 1
            y = slice_along_axis(y, 0, last, axis=axis)
        
        #add previous overlap to convolved and update overlap
        slices = [slice(None)] * subarr.ndim
        slices[axis] = slice(0, M - 1)
        y[tuple(slices)] += overlap
        overlap = new_overlap
        
        #apply the boundary mode to first and last segments
        if segment_num == 0 or segment_num == nsegments - 1:
            y = _oa_mode(y, segment_num, len(win), axis, mode)
        
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

@as_producer
def polyphase_resample(pro, L, M, fs, chunksize, axis=-1, **kwargs):
    """Resamples an array or producer of arrays by a rational factor (L/M)
    using the polyphase decomposition.

    Args:
        pro: A producer of ndarrays
            The data producer to be resampled.
        L: int
            The expansion factor. L-1 interpolated values will be inserted
            between consecutive samples along axis.
        M: int
            The decimation factor describing which Mth samples of produced 
            data survive decimation. (E.g. M=10 -> every 10th survives)
        fs: int
            The sampling rate of produced data in Hz.
        chunksize: int
            The number of samples to hold in memory during upsampling.
            This method will require ~ 3 times chunksize in memory.
        axis: int
            The axis of produced data along which downsampling will occur.
        kwargs:
            Any valid keyword for a Kaiser lowpass filter. The default 
            values for combined antialiasing & interpolation filter are:

                fstop: int
                    The stop band edge frequency. 
                    Defaults to fs // max(L,M).
                fpass: int
                    The pass band edge frequency. Must be less than fstop.
                    Defaults to fstop - fstop // 10.
                gpass: int
                    The pass band attenuation in dB. Defaults to a max loss
                    in the passband of 1 dB ~ 11% amplitude loss.
                gstop: int
                    The max attenuation in the stop band in dB. Defaults to
                    40 dB or 99%  amplitude attenuation.

    Returns: a producer of resampled data. The chunksize of the yielded
             arrays along axis will be the nearest multiple of M closest to 
             the supplied chunksize (e.g. if M=3 and chunksize=1000 the 
             yielded chunksize will be 1002 since 1002 % 3 == 0).
    """

    if M >= pro.shape[axis]:
        msg = 'Decimation factor must M={} be < pro.shape[{}] = {}'
        raise ValueError(msg.format(M, axis, pro.shape[axis]))

    # pathological case: pro has  < 3 chunks  -> autoreduce csize
    csize = chunksize
    if csize > pro.shape[axis] // 3:
        csize = pro.shape[axis]  // 3

    # kaiser antialiasing & interpolation filter coeffecients
    fstop = kwargs.pop('fstop', fs // max(L, M))
    fpass = kwargs.pop('fpass', fstop - fstop // 10)
    gpass, gstop = kwargs.pop('gpass', 1), kwargs.pop('gstop', 40)
    h = Kaiser(fpass, fstop, fs, gpass, gstop).coeffs

    # ensure decimation of each produced array is integer samples
    if csize % M > 0:
        csize = int(np.ceil(csize / M) * M)

    # iterators for prior, current and next chunks of data
    x = producer(pro, csize, axis)
    y = producer(pro, csize, axis)
    z = producer(pro, csize, axis)
    iprior, icurrent, inext = (iter(pro) for pro in [x,y,z])

    # num pts to append left & right on axis to cover convolve overhang
    # must be divisible by M to ensure int slicing after resample.
    overhang = int(np.ceil((len(h) - 1) / M) * M)

    # initialize left/right pads for first data section
    left_shape = list(pro.shape)
    left_shape[axis] = overhang
    left = np.zeros(left_shape)
    # advance inext twice to get right pads
    next(inext)
    right = slice_along_axis(next(inext), 0,  overhang, axis=axis)

    # compute the first resampled chunk
    current = next(icurrent)
    padded = np.concatenate((left, current, right), axis=axis)
    resampled = sps.resample_poly(padded, up=L, down=M, axis=axis, window=h)

    # remove result points computed from pads
    a, b = int(overhang * L / M), -int(overhang * L / M)
    yield slice_along_axis(resampled, a, b, axis=axis)

    # resample remaining chunks
    cnt = z.shape[axis] // csize + bool(z.shape[axis] % csize) - 1
    for n, (last, curr, nxt) in enumerate(zip(iprior, icurrent, inext), 1):

        # build left and right pads for current
        left = slice_along_axis(last, -overhang, axis=axis)
        
        if n < cnt - 1:
            right = slice_along_axis(nxt, 0, overhang, axis=axis)
        else:
            # at cnt-1 chunks concantenate next to current
            curr = np.concatenate((curr, nxt), axis=axis) 
            right = np.zeros(left.shape)

        padded = np.concatenate((left, curr, right), axis=axis)
        resampled = sps.resample_poly(padded, L, M, axis=axis, window=h)
        yield slice_along_axis(resampled, a, b, axis=axis)


def periodogram(arr, fs, nfft=None, window='hann', axis=-1, 
                detrend='constant', scaling='density'):
    """Estimates the power spectrum of an ndarray using the windowed
    periodogram method.

    Args:
        arr: ndarray
            An array of values to estimate the PS or PSD along axis.
            This array is assumed to be real-valued (Hermetian symmetric)
        fs: int
            The sampling rate of the values in arr.
        nfft: int
            The number of frequencies between 0 and fs used to construct
            the Discrete Fourier Transform. If None, nfft will match the 
            length of the arr. If nfft is smaller than arr along axis, the
            array is cropped. If nfft is larger than arr along axis, the 
            array is zero padded. The returned frequencies will be nfft/2
            since this method returns only positive frequencies.
        window: str
            A scipy signal module window function. Please see references
            for all available windows.
        axis: int
            Axis along which the power spectrum will be estimated.
        detrend: str
            The type of detrending to apply to data before computing
            estimate. Options are 'constant' and 'linear'. If constant, the
            mean of the data is removed before the PS/PSD is estimated. If
            linear, the linear trend in the data array along axis is removed
            before the estimate.
        scaling: str
            A string for determining the normalization of the estimate. If
            'spectrum', the estimate will have units V**2 and be referred to
            as the power spectral estimate. If 'density' the estimate will
            have units V**2 / Hz and is referred to as the power spectral
            density estimate.
            
    Returns:
        A 1-D array of length NFFT/2 of postive frequencies at which the 
        estimate was computed.

        An ndarray of power spectral (density) estimates the same shape as
        array except along axis which we have length NFFT/2.

    References:
        Schuster, Arthur (January 1898). "On the investigation of hidden 
        periodicities with application to a supposed 26 day period of 
        meteorological phenomena". Terrestrial Magnetism. 3 (1): 13–41

        Scipy windows:
        https://docs.scipy.org/doc/scipy/reference/signal.windows.html
    """

    nsamples = arr.shape[axis]
    nfft = nsamples if not nfft else int(nfft)

    if nfft < nsamples:
        # crop arr before detrending & windowing; see rfft crop
        arr = slice_along_axis(arr, 0, nfft, axis=-1)
    
    # detrend the array
    arr = sps.detrend(arr, axis=axis, type=detrend)

    # fetch and apply window
    coeffs = sps.get_window(window, arr.shape[axis])
    arr = arr * coeffs

    # compute real DFT. Zeropad for nfft > nsamples is automatic
    arr = np.fft.rfft(arr, nfft, axis=axis)
    freqs = np.fft.rfftfreq(nfft, d=1/fs)

    # scale using weighted mean of window values
    if scaling == 'spectrum':
        norm = 1 / np.sum(coeffs)**2

    elif scaling == 'density':
        norm = 1 / (fs * np.sum(coeffs**2))
    
    else:
        msg = 'Unknown scaling: {}'
        raise ValueError(msg.format(scaling))
    arr = (np.real(arr)**2 + np.imag(arr)**2) * norm

    # since real FFT -> double for uncomputed negative freqs.
    slicer = [slice(None)] * arr.ndim
    if nfft % 2:
        # k=0 dft sample is not pos. or neg so not doubled
        slicer[axis] = slice(1, None)
    
    else:
        # last k=nfft/2 is not in the dft since nfft is odd
        slicer[axis] = slice(1, -1)
    
    arr[tuple(slicer)] *= 2

    return freqs, arr


def _welch(pro, fs, nperseg, window, axis, csize=100, **kwargs):
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


    """
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

    """
    x = np.concatenate([arr for arr in pro], axis=-1)
    t0 = time.perf_counter()
    sp_f, sp_pxx = sps.welch(x, fs=5000, nperseg=16384, window='hann', axis=-1)
    print('sp welch in {} s'.format(time.perf_counter() - t0))
    """

    """
    plt.plot(f, pxx[0], label='os result')
    """

    """
    plt.plot(f, sp_pxx[0], label='sp result')
    plt.plot(f, (pxx[0] - sp_pxx[0]), label='residual')
    plt.legend()
    plt.show()
    """

    x = np.array(range(7))
    y = np.array([0, 1, .5, 2, 0])
    z = np.convolve(x,y)
    r = convolve_slicer(z, x.shape, y.shape, axis=-1, mode='valid')
    o = np.convolve(x,y, mode='valid')
    assert np.allclose(r, o)
