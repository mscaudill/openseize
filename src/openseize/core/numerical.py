import numpy as np
import itertools
from functools import partial
import scipy.signal as sps

from openseize import producer
from openseize.core.producer import Producer, as_producer, pad_producer 
from openseize.core.queues import FIFOArray
from openseize.core.arraytools import (pad_along_axis, slice_along_axis,
        multiply_along_axis, split_along_axis)


def optimal_nffts(arr):
    """Estimates the number of FFT points for the overlap-add convolution.

    Args:
        arr: 1-D array
            The window which will be convolved across a larger ndarry.

    This is an approximation for the value that minimizes oa cost func:

        oa_cost = Nx * NFFT * (log2(NFFT) + 1) / (NFFT - len(arr) + 1) 
    
        Nx is length of larger array along convolve axis
        NFFT is the number of FFT points to be estimated
        arr is the length of the convolving window.
        see https://en.wikipedia.org/wiki/Overlap-add_method

    Returns: Integer number of NFFT pts.
    """

    return int(8 * 2 ** np.ceil(np.log2(len(arr))))


def convolved_shape(shape1, shape2, mode, axis):
    """Computes the shape of the convolution of two ndarrays along axis.
    
    Args:
        shape1: tuple
            Shape of the first input used to construct the convolved arr.
        shape2: tuple
            Shape of the second input used to construct the convolved arr.
        mode: str one of 'full', 'same', or 'valid'
            A string defining the boundary handling mode. These modes
            are the same as numpy's np.convolve mode argument.
        axis: int
            The convolution axis in arr.

    Returns: A tuple for the shape of the resulting convolution.
    """

    m, n = shape1[axis], shape2[axis]
    p, q = max(m,n), min(m,n)
    
    # find array with largest ndims
    outshape = sorted([list(shape1), list(shape2)], key=len)[-1]

    if mode == 'full':
        outshape[axis] = m + n - 1
    
    elif mode == 'same':
        outshape[axis] = p

    elif mode == 'valid':
        outshape[axis] = m + n - 1 - 2 * (q-1)

    return tuple(outshape)


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


def _oa_boundary(arr, window, side, axis, mode):
    """Applies the numpy convolve mode to first or last segment yielded from
    oaconvolve.

    Args:
        arr: ndarray
            The first or last array of values from the oaconvolve producer.
        window: 1-D array
            The window that was convolved across arr.
        side: str
            One of 'left', 'right' indicating if the array is the first or
            last segment from oaconvolve.
        axis: int
            The axis of arr along which convolution was performed.
        mode: str
            A numpy convolve mode -- one of 'full', 'same', 'valid'.
    
    Returns: An ndarray with boundary mode applied
    """

    ns = arr.shape[axis]
    lw = len(window)
    
    #dict of slices to apply to axis for 0th & last segment for each mode
    cuts = {'full': {'left': slice(None), 
                     'right': slice(None)},

            'same': {'left': slice((lw - 1) // 2, None), 
                     'right': slice(None, ns - int(np.ceil((lw - 1) / 2)))},

            'valid': {'left': slice(lw - 1, None), 
                      'right': slice(None, ns - lw + 1)}}

    #apply slices
    slices = [slice(None)] * arr.ndim
    slices[axis] = cuts[mode][side]
    return arr[tuple(slices)]


def oaconvolve(pro, window, axis, mode, nfft_factor=32):
    """Performs overlap-add circular convolution of a producer of 
    ndarrays with a 1-dimensional window.

    Args:
        pro: producer of ndarrays
            The data to be convolved.
        window: 1-D array            
            A 1-D window to convolve across data.
        axis: int                 
            The axis of data along which window should be convolved.
        mode: str
            A convolution mode matching one of 'full', 'same', 'valid'.
            These modes are identical to numpy convolve modes.
        nfft_factor: int
            A power of 2 that multiplies the optimal number of nffts
            used by this algorithms circular convolution. Increasing this
            number by factors of 2 may increase the speed of this algorithm.
            Defaults to 32 (see Implementation Notes)

    Numpy & Scipy implementations of oaconvolve require that the input be an
    in-memory array. Openseize utilizes a producer making it suitable for
    convolving 1-D arrays across data that may not fit into memory.

    Implementation Notes: 
    The overlap-add convolve method processes small segments O(len(window)).
    The producer then may have to perform many small reads if its
    data source is a file. To avoid this, oaconvolve uses the FIFOArray
    queue as a cache to hold larger arrays of the data. These larger cached
    arrays are then iterated over in chunks of the small segments. This
    complicates the code's structure but gives a large performance boost.

    The additional nfft_factor effectively increases the amount of samples
    that the algorithm will convolve at once. By increasing the nffts above
    the optimal value for the FFT algorithm, we allow oaconvolve to convolve
    more data at a single time which reduces the data fetches. This improves
    performance up until the processors cache size is reached. The value
    should be decreased if the window is large and increased if the window
    is small to improve performance.

    Returns: A generator of convolved ndarrays.
    """

    # compute the near optimal nfft number and FFT of window
    nfft = optimal_nffts(window) * nfft_factor

    # fallback to optimal nffts if nfft_factor makes nfft > data size.
    if nfft - len(window) + 1 > pro.shape[axis]:
        nfft = optimal_nffts(window)

    wlen = len(window)
    H = np.fft.rfft(window, nfft)

    # set the step size based on optimal nfft and wlen
    step = nfft - wlen + 1

    nsegments = int(np.ceil(pro.shape[axis] / step))

    # create the wlen-1 samples overlap
    overlap_shape = list(pro.shape)
    overlap_shape[axis] = wlen - 1
    overlap = np.zeros(overlap_shape)

    # FIFOArray holding chunksize arrays but yielding step size arrays
    fifo = FIFOArray(step, axis)

    # Helper funcs
    def _cconvolve(x, H, nfft, wlen, axis):
        """Circularly convolves a data segment, x, with H, the 
        fft of the window of len wlen along axis."""
        
        # pad with wlen-1 zeros for overlap & FFT
        x = pad_along_axis(x, [0, wlen - 1], axis=axis)
        xf = np.fft.rfft(x, nfft, axis=axis)
            
        # take product with window in freq. domain
        product = multiply_along_axis(xf, H, axis=axis)

        # back transform to sample domain and return
        return  np.fft.irfft(product, axis=axis).real

    def _add_overlap(y, overlap, wlen, axis):
        """Adds overlap to first wlen-1 samples of yth segment 
        along axis."""

        slicer = [slice(None)] * y.ndim
        slicer[axis] = slice(0, wlen-1)
        y[tuple(slicer)] += overlap

        return y

    segment = 0
    for arr in pro:

        fifo.put(arr)

        while fifo.qsize() > step:
           
            # get data segment & cicularly convolve
            arr = fifo.get()
            z = _cconvolve(arr, H, nfft, wlen, axis)
            
            #split segment & next overlap
            y, new_overlap = split_along_axis(z, step, axis=axis)

            # add previous overlap & update overlap
            y = _add_overlap(y, overlap, wlen, axis)
            overlap = new_overlap

            #apply the boundary mode to first and last segments
            if segment == 0:
                y = _oa_boundary(y, window, 'left', axis, mode)
        
            #update segment
            segment += 1
        
            yield y
        
        else:
            # put new data into fifo
            continue
    else:
        
        if not fifo.empty():
           
            # get all remaining in queue & circularly convolve
            arr = fifo.queue
            z = _cconvolve(arr, H, nfft, wlen, axis)

            # last segment has wlen - 1 overhang
            last = arr.shape[axis] + wlen - 1
            y = slice_along_axis(z, 0, last, axis=axis)

            #add previous overlap
            y = _add_overlap(y, overlap, wlen, axis)

            yield _oa_boundary(y, window, 'right', axis, mode)


@as_producer
def sosfilt(pro, sos, axis, zi=None):
    """Batch applies a forward second-order-section fmt filter to a 
    producer of numpy arrays.

    Args:
        pro: producer of ndarrays
            A producer of chunksize ndarrays along axis to filter.
        sos: 2-D array
            An n-sections x 6 array of numerator & denominator coeffs of the
            transfer function of an IIR filter.
        axis: int
            The axis along which to apply the filter in chunksize batches
        zi: ndarray
            Initial conditions of the filter. This is an n-sections x 
            (...,2,...) array where (...,2...) has the same shape as pro 
            but with 2 along axis. This 2 is because biquad section of the
            sos fmt has a delay of 2 along axis. For further details see 
            sosfilt_zi in scipy's signal module. Default is None
            which sets the initial nsections of filtered values to 0.

    Returns: A producer of filtered values of len chunksize along axis.
    """

    # set initial conditions for filter (see scipy sosfilt; zi)
    shape = list(pro.shape)
    shape[axis] = 2
    z = np.zeros((sos.shape[0], *shape)) if zi is None else zi
    
    # compute filter values & store current initial conditions 
    for subarr in pro:
        
        y, z = sps.sosfilt(sos, subarr, axis=axis, zi=z)
        yield y


@as_producer
def sosfiltfilt(pro, sos, axis, **kwargs):
    """Batch applies a forward-backward second order section fmt filter to
    a producer of numpy arrays.

    Args:
        pro: producer of ndarrays
            A producer of ndarrays of shape chunksize along axis to filter.
        sos: 2-D array
            An n-sections x 6 array of numerator & denominator coeffs of the
            transfer function of an IIR filter.
        axis: int
            The axis along which to apply the filter in chunksize batches

    Returns: a producer of forward-backward filtered values of len 
             chunksize along axis

    Notes: 
        1. This iterative algorithm is not nearly as efficient as working on
           the full array. For each forward/backward filtered sub arr in
           producer we must perform a forward/backward filter of two
           subarrs in producer. This gives us the correct initial conditions
           at the boundaries of the arrays. It is one reason why FIR filters
           should be preferred in Openseize.

        2. Since the filter is a forward/backward filter the initial 
           conditions are handled automatically.

        3. This algorithm does not allow for boundary padding like scipy
           sosfiltfilt. There is exact agreement b/w openseize and scipy
           only when scipys sosfiltfilt is called with padtype=None. All
           other padtypes will show slight differences at the leftmost and
           rightmost samples along axis.
    """

    # get initial value from producer
    subarr = next(iter(pro))
    x0 = slice_along_axis(subarr, 0, 1, axis=axis) 

    # build steady state initial condition
    zi = sps.sosfilt_zi(sos) #nsections x 2
    # reshape zi so zi * x0 broadcast to correct shape for zi param
    s = [1] * len(pro.shape)
    s[axis] = 2
    zi = np.reshape(zi, (sos.shape[0], *s)) #nsections,1,2

    # build a generators of forward filtered with one advanced
    a_gen = iter(sosfilt(pro, sos, axis, zi=zi*x0))  
    b_gen = iter(sosfilt(pro, sos, axis, zi=zi*x0))
    next(b_gen)

    n = int(np.ceil(pro.shape[axis] / pro.chunksize))
    for idx, a in enumerate(a_gen, 1):

        if idx < n:

            b = next(b_gen)
            # for reverse filter, use final delay values from flipped
            # advanced 'b' arr as initial values for current flipped arr. 
            bflipped = np.flip(b, axis=axis)
            b_0 = slice_along_axis(bflipped, 0, 1, axis=axis)
            _, zf = sps.sosfilt(sos, bflipped, axis=axis, zi=zi*b_0)

            aflipped = np.flip(a, axis=axis)
            rfilt, _ = sps.sosfilt(sos, aflipped, axis=axis, zi=zf)
            yield np.flip(rfilt, axis=axis)

        else:
            
            # for last segment the initial condition is last sample ss
            aflipped = np.flip(a, axis=axis)
            a0 = slice_along_axis(aflipped, 0, 1, axis=axis)
            rfilt, _ = sps.sosfilt(sos, aflipped, axis=axis, zi=zi*a0)
            yield np.flip(rfilt, axis=axis)


@as_producer
def lfilter(pro, coeffs, axis, zi=None):
    """Batch appliies a forward transfer function fmt (b,a) filter to
    a producer of numpy arrays.

    Args:
        pro: producer of ndarrays
            A producer of ndarrays of shape chunksize along axis to filter.
        coeffs: tuple
            A tuple of numerator, denominator coefficient arrays of the
            transfer function fmt (b,a).
        axis: int
            The axis along which to apply the filter in chunksize batches
        zi: ndarray
           The initial output values of the filtered data. If None
           (default), the initial values are zeros. Please see scipy
           signal lfilter for more details.
    
    Returns: A producer of filtered values of len chunksize along axis.    
    """

    b, a = coeffs

    # set initial conditions of the filters output
    shape = list(pro.shape)
    shape[axis] = int(max(len(b), len(a)) - 1)
    z = np.zeros(shape) if zi is None else zi

    # compute filter values & store current initial conditions
    for subarr in pro:
        
        y, z = sps.lfilter(b, a, subarr, axis=axis, zi=z)
        yield y


@as_producer
def filtfilt(pro, coeffs, axis, **kwargs):
    """Batch applies a forward-backward filter in transfer func fmt (b,a)
    to a producer of numpy arrays.

    Args:
        pro: producer of ndarrays
            A producer of ndarrays to filter.
        coeffs: tuple
            A tuple of numerator, denominator coefficient arrays of the
            transfer function fmt (b,a).
        axis: int
            The axis along which to apply the filter in chunksize batches

    Returns: A producer of filtered values of len chunksize along axis.

    Notes:
        1. This iterative algorithm is not nearly as efficient as working on
           the full array. For each forward/backward filtered sub arr in
           producer we must perform a forward/backward filter of two
           subarrs in producer. This gives us the correct initial conditions
           at the boundaries of the arrays. It is one reason why FIR filters
           should be preferred in Openseize.

        2. Since the filter is a forward/backward filter the initial 
           conditions are handled automatically.

        3. This algorithm does not allow for boundary padding like scipy
           filtfilt. There is exact agreement b/w openseize and scipy
           only when scipys filtfilt is called with padtype=None. All
           other padtypes will show slight differences at the leftmost and
           rightmost samples along axis.
    """

    # get initial value from producer
    x0 = slice_along_axis(next(iter(pro)), 0, 1, axis=axis)

    # get steady state initial conditions
    zi = sps.lfilter_zi(*coeffs) 
    # reshape zi so zi * x0 broadcast to correct shape for zi param
    s = [1] * len(pro.shape)
    s[axis] = zi.size
    zi = np.reshape(zi, s)

    # build generators of forward filtered with one advanced
    x_gen = iter(lfilter(pro, coeffs, axis, zi=zi*x0))
    y_gen = iter(lfilter(pro, coeffs, axis, zi=zi*x0))
    next(y_gen)

    n = int(np.ceil(pro.shape[axis] / pro.chunksize))
    for idx, x in enumerate(x_gen, 1):

        if idx < n:

            y = next(y_gen)
            # for reverse filter, use final delay values from flipped
            # advanced 'y' arr as initial values for current flipped x
            yflipped = np.flip(y, axis=axis)
            y0 = slice_along_axis(yflipped, 0, 1, axis=axis)
            _, zf = sps.lfilter(*coeffs, yflipped, axis=axis, zi=zi*y0)

            xflipped = np.flip(x, axis=axis)
            rfilt, _ = sps.lfilter(*coeffs, xflipped, axis=axis, zi=zf)
            yield np.flip(rfilt, axis=axis)
        
        else:

            # for last segment the initial condition is last sample ss
            xflipped = np.flip(x, axis=axis)
            x0 = slice_along_axis(xflipped, 0, 1, axis=axis)
            rfilt, _ = sps.lfilter(*coeffs, xflipped, axis=axis, zi=zi*x0)
            yield np.flip(rfilt, axis=axis)


def polyphase_resample(pro, L, M, fs, fir, axis, **kwargs):
    """Resamples an array or producer of arrays by a rational factor (L/M)
    using the polyphase decomposition.

    Args:
        pro: A producer of ndarrays
            The data producer of arrays of shape chunksize along axis to 
            be resampled. This method will require ~3 times chunksize in
            memory. 
        L: int
            The expansion factor. L-1 interpolated values will be inserted
            between consecutive samples along axis.
        M: int
            The decimation factor describing which Mth samples of produced 
            data survive decimation. (E.g. M=10 -> every 10th survives)
        fs: int
            The sampling rate of produced data in Hz.
        fir: FIR filter
            An openseize fir filter class
        axis: int
            The axis of produced data along which downsampling will occur.
        kwargs:
            Any valid keyword for a Kaiser lowpass filter. The default 
            values for combined antialiasing & interpolation filter are:

                fstop: int
                    The stop band edge frequency. 
                    Defaults to cutoff + cutoff / 10 where cutoff = 
                    fs //(2 * max(L,M)).
                fpass: int
                    The pass band edge frequency. Must be less than fstop.
                    Defaults to cutoff - cutoff / 10 where cutoff = fs //
                    (2 * max(L,M)).
                gpass: int
                    The pass band attenuation in dB. Defaults to a max loss
                    in the passband of 0.1 dB ~ 1.1% amplitude loss.
                gstop: int
                    The max attenuation in the stop band in dB. Defaults to
                    40 dB or 99%  amplitude attenuation.

    Returns: a generator of resampled data. The chunksize of the yielded
             arrays along axis will be the nearest multiple of M closest to 
             the pro.chunksize (e.g. if M=3 and chunksize=1000 the 
             yielded chunksize will be 1002 since 1002 % 3 == 0).
    """

    if M >= pro.shape[axis]:
        msg = 'Decimation factor must M={} be < pro.shape[{}] = {}'
        raise ValueError(msg.format(M, axis, pro.shape[axis]))

    # pathological case: pro has  < 3 chunks  -> autoreduce csize
    csize = pro.chunksize
    if csize > pro.shape[axis] // 3:
        csize = pro.shape[axis]  // 3

    # kaiser antialiasing & interpolation filter coeffecients
    cutoff = fs / (2*max(L, M))
    fstop = kwargs.pop('fstop', cutoff + cutoff / 10)
    fpass = kwargs.pop('fpass', cutoff - cutoff / 10)
    gpass, gstop = kwargs.pop('gpass', 0.1), kwargs.pop('gstop', 40)
    h = fir(fpass, fstop, fs, gpass, gstop).coeffs

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


def modified_dft(arr, fs, nfft, window, axis, detrend, scaling):
    """Returns the windowed Discrete Fourier Transform of a real signal.

    Args:
        arr: ndarray
            An array of values to estimate the DFT along axis.
            This array is assumed to be real-valued (Hermetian symmetric)
        fs: int
            The sampling rate of the values in arr.
        nfft: int
            The number of frequencies between 0 and fs used to construct
            the DFT. If None, nfft will match the length of the arr. If 
            nfft is smaller than arr along axis, the array is cropped. If
            nfft is larger than arr along axis, the array is zero padded.
            The returned frequencies will be nfft//2 + 1 since this method
            returns only positive frequencies.
        window: str
            A scipy signal module window function. Please see references
            for all available windows.
        axis: int
            Axis along which the DFT will be computed.
        detrend: str
            The type of detrending to apply to data before computing
            DFT. Options are 'constant' and 'linear'. If constant, the
            mean of the data is removed before the DFT is computed. If
            linear, the linear trend in the data array along axis is 
            removed.
        scaling: str
            A string for determining the normalization of the DFT. If
            'spectrum', the DFT * np.conjugate(DFT) will have units V**2.
            If 'density' the DFT * np.conjugate(DFT) will have units 
            V**2 / Hz.

    Returns:
        A 1-D array of length nfft//2 + 1 of positive frequencies at which
        the DFT was computed.

        An ndarray of DFT the same shape as array except along axis which
        will have length nfft//2 + 1

    References:
        Shiavi, R. (2007). Introduction to Applied Statistical Signal 
        Analysis : Guide to Biomedical and Electrical Engineering 
        Applications. 3rd ed.

        Scipy windows:
        https://docs.scipy.org/doc/scipy/reference/signal.windows.html
    """

    nsamples = arr.shape[axis]

    if nfft < nsamples:
        # crop arr before detrending & windowing; see rfft crop
        arr = slice_along_axis(arr, 0, nfft, axis=-1)

    # detrend the array
    arr = sps.detrend(arr, axis=axis, type=detrend)

    # fetch and apply window
    coeffs = sps.get_window(window, arr.shape[axis])
    arr = multiply_along_axis(arr, coeffs, axis=axis)

    # compute real DFT. Zeropad for nfft > nsamples is automatic
    # rfft uses 'backward' norm default which is no norm on rfft
    arr = np.fft.rfft(arr, nfft, axis=axis)
    freqs = np.fft.rfftfreq(nfft, d=1/fs)

    # scale using weighted mean of window values
    if scaling == 'spectrum':
        norm = 1 / np.sum(coeffs)**2

    elif scaling == 'density':
        #process loss Shiavi Eqn 7.54
        norm = 1 / (fs * np.sum(coeffs**2))
    
    else:
        msg = 'Unknown scaling: {}'
        raise ValueError(msg.format(scaling))
   
    # before conjugate multiplication unlike scipy
    # see _spectral_helper lines 1808 an 1842.
    arr *= np.sqrt(norm)

    return freqs, arr


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
            array is zero padded. The returned frequencies will be 
            nfft//2 + 1 since this method returns only positive frequencies.
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
        A 1-D array of length NFFT/2 of positive frequencies at which the 
        estimate was computed.

        An ndarray of power spectral (density) estimates the same shape as
        array except along axis which we have length nfft//2 + 1.

    References:
        Schuster, Arthur (January 1898). "On the investigation of hidden 
        periodicities with application to a supposed 26 day period of 
        meteorological phenomena". Terrestrial Magnetism. 3 (1): 13–41

        Shiavi, R. (2007). Introduction to Applied Statistical Signal 
        Analysis : Guide to Biomedical and Electrical Engineering 
        Applications. 3rd ed.

        Scipy windows:
        https://docs.scipy.org/doc/scipy/reference/signal.windows.html
    """

    nfft = arr.shape[axis] if not nfft else int(nfft)
    
    # compute modified DFT & take modulus to get power spectrum
    freqs, arr = modified_dft(arr, fs, nfft, window, axis, detrend, scaling)
    arr = np.real(arr)**2 + np.imag(arr)**2 

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


def _spectra_estimatives(pro, fs, nfft, window, overlap, axis, detrend,
                         scaling, func, **kwargs):
    """Iteratively estimates the power spectrum or modified DFT for each
    nfft segment in a producer.

    This generator yields an estimate for each nfft segment. It should not
    be called externally.

    Args:
        func: function
            A function returning a spectral estimate for a window of data.
            E.g. periodogram, modified_dft
        kwargs: unused kwargs
            Future support for additional spectral estimate functions.
    """

    # num overlap points & shift between successive nfft segments
    noverlap = int(nfft * overlap)
    stride = nfft - noverlap

    # use FIFO to cache & release nfft & stride  num. samples respectively
    fifo = FIFOArray(chunksize=stride, axis=axis)
    for n, arr in enumerate(pro):

        # yield nfft sized estimates while fifo has >= nfft samples
        while fifo.qsize() >= nfft:
            
            # slice nfft samples to compute estimate
            x = slice_along_axis(fifo.queue, 0, nfft, axis=axis)
            f, y = func(x, fs, nfft, window, axis, detrend, scaling)
            
            # release stride samples leaving nover in FIFO
            fifo.get()
            yield y
        
        else:
            
            # provide fifo with more samples
            fifo.put(arr)
            continue
    
    else:
        
        # exhaust the fifo
        while fifo.qsize() >= nfft:
            
            x = slice_along_axis(fifo.queue, 0, nfft, axis=axis)
            f, y = func(x, fs, nfft, window, axis, detrend, scaling)
            fifo.get() 
            yield y


def welch(pro, fs, nfft, window, overlap, axis, detrend, scaling):
    """Iteratively estimates the power spectrum using Welch's method.

    Welch's method divides data into overlapping segments and computes the
    modified periodograms for each segment. Usually these segments are
    average over to estimate the PSD under the assumption that the data is
    stationary (i.e. the spectral content across segments is similar.)
    Unlike scipy, this method does not assert that the data is an in-memory
    array.

    Args:
        pro: A producer of ndarrays
            A data producer whose power spectral is to be estimated.
        fs: int
            The sampling rate of the produced data.
        nfft: int
            The number of frequencies in the interval [0, fs) to use to
            estimate the power spectra. This determines the frequency
            resolution of the estimate since resolution = fs / nfft.
        window: str
            A string name for a scipy window to be applied to each data
            segment before computing the periodogram of that segment. For
            a full list of windows see scipy.signal.windows.
        overlap: float
            A percentage in [0, 1) of the data segment that should overlap
            with the next data segment. If 0 this estimate is equivalent to
            Bartletts method (2)
        axis: int
            The sample axis of the producer. The estimate will be carried
            out along this axis.
        detrend: str either 'constant' or 'linear'
            A string indicating whether to subtract the mean ('constant') or
            subtract a linear fit ('linear') from each segment prior to
            computing the estimate for a segment.
        scaling: str either 'spectrum' or 'density'
            Determines the normalization to apply to the estimate. If
            'spectrum' the estimate will have units V**2 and if 'density'
            V**2 / Hz.

    Returns:
        A tuple containing a 1-D array of frequencies of length nfft//2 + 1
        and a producer of PS(D) estimates for each overlapped segment.

    Notes:
        Scipy allows for the segment length and number of DFT points (nfft)
        to be different. This allows for interpolation of frequencies. Given
        that EEG data typically has many samples, openseize locks the
        segment length to the nfft amount (i.e. no interpolation). Finer
        resolutions of the estimate will require longer data segments.

        Scipy welch drops the last segment of data if the number of points in 
        the segment is less than nfft. Openseize welch follows the same 
        convention. 

        Lastly, Openseize assumes the produced data is real-valued. This is
        appropriate for all EEG data. If you are calling this method on
        complex data, the imaginary part will be dropped.

    References:
        (1) P. Welch, "The use of the fast Fourier transform for the 
        estimation of power spectra: A method based on time averaging over 
        short, modified periodograms", IEEE Trans. Audio Electroacoust. vol.
        15, pp. 70-73, 1967

        (2) M.S. Bartlett, "Periodogram Analysis and Continuous Spectra", 
        Biometrika, vol. 37, pp. 1-16, 1950.

        (3) B. Porat, "A Course In Digital Signal Processing" Chapters 4 &
        13. Wiley and Sons 1997.
    """

    # build the welch generating function
    genfunc = partial(_spectra_estimatives, pro, fs, nfft, window, overlap, 
                      axis, detrend, scaling, func=periodogram)

    # obtain the positive freqs.
    freqs = np.fft.rfftfreq(nfft, 1/fs)

    # num. segments that fit into pro samples of len nfft with % overlap
    nsegs = int((pro.shape[axis] - nfft) // (nfft * (1-overlap)) + 1)
    shape = list(pro.shape)
    shape[axis] = nsegs

    # return producer from welch gen func with each yielded 
    result = producer(genfunc, chunksize=len(freqs), axis=axis, shape=shape)
    return freqs, result


def stft(pro, fs, nfft, window, overlap, axis, detrend, scaling, boundary,
         padded):
    """Estimates the Discrete Short-time Fourier Transform of a real signal.

    STFT breaks the signal into overlapping segments and computes
    a windowed Discrete Fourier Transform for each segment. This is a
    complex sequence X(freq, time). The spectrogram is then 
    np.abs(X(freq, time))**2.

    Args:
        pro: A producer of ndarrays
            A data producer whose STFT is to be estimated.
        fs: int
            The sampling rate of the produced data.
        nfft: int
            The number of frequencies in the interval [0, fs) to use to
            estimate the spectra in each segment. This determines the 
            frequency resolution of the estimate since 
            resolution = fs / nfft.
        window: str
            A string name for a scipy window to be applied to each data
            segment before computing the periodogram of that segment. For
            a full list of windows see scipy.signal.windows.
        overlap: float
            A percentage in [0, 1) of the data segment that should overlap
            with the next data segment. In order for the STFT to be
            invertible. The overlap amount needs to equally weight all
            samples in pro. This is called the "constant overlap-add" COLA
            constraint. It is window dependent. If you intend to invert this
            STFT (synthesis) you  will need to verify this constraint is met
            using scipy.signal.check_COLA. 
        axis: int
            The sample axis of the producer. The estimate will be carried
            out along this axis.
        detrend: str either 'constant' or 'linear'
            A string indicating whether to subtract the mean ('constant') or
            subtract a linear fit ('linear') from each segment prior to
            computing the estimate for a segment.
        scaling: str either 'spectrum' or 'density'
            Determines the normalization to apply to the estimate for each
            segment. If 'spectrum', then np.abs(X)**2 is the magnitude 
            spectrum for each segment. If density np.abs(X)**2 is the
            density spectrum and may be integrated over to give the total 
            power for a segment.
        boundary: bool
            A boolean indicating if the first and last segments should be
            extended with zeros so that the first/last samples ar centered
            at nfft//2.  This allows for inversion of the first/last input 
            points for windows whose first & last values are 0. Unlike 
            scipy, openseize only allows for zero extensions since this has
            the clear interpretation of a zero-pad interpolation of the
            frequencies in the DFT.
        padded: bool
            Specifies if the last array of the producer should be padded
            with zeros so that a whole number of overlapped nfft segments
            fit into the signal length. This enables inversion of the stft
            since the entire signal is used. This is in contrast with Welch
            which drops the last segment if shorter than nfft - noverlap.

    Returns:
        f: 1-D array of frequencies
        t: 1-D array of segment times
        X: Producer of stft estimates for each segment. Each yielded array
           has length nfft - nfft * overlap along axis.

    Notes:
        Scipy allows for the segment length and number of DFT points (nfft)
        to be different. This allows for interpolation of frequencies. Given
        that EEG data typically has many samples, openseize locks the
        segment length to the nfft amount (i.e. no interpolation). Finer
        resolutions of the estimate will require longer data segments.

        Openseize assumes the produced data is real-valued. This is
        appropriate for all EEG data. If you are calling this method on
        complex data, the imaginary part will be dropped.

    
    References:
        (1) Shiavi, R. (2007). Introduction to Applied Statistical Signal 
        Analysis : Guide to Biomedical and Electrical Engineering 
        Applications. 3rd ed.

        (2) B. Porat, "A Course In Digital Signal Processing" Chapters 4 &
        13. Wiley and Sons 1997.
    """

    # num overlap points & shift between successive nfft segments
    noverlap = int(nfft * overlap)
    stride = nfft - noverlap

    # stft boundary & padding options
    data = pro
    if boundary:
        
        # center first & last segments by padding producer
        data = pad_producer(data, nfft//2, value=0)

    if padded:
        
        nsamples = pro.shape[axis]
        # pad w/ stride if samples not divisible by stride
        amt = stride if nsamples % stride else 0
        data = pad_producer(data, [0, amt], value=0)

    # build the stft generating function
    genfunc = partial(_spectra_estimatives, data, fs, nfft, window, overlap, 
                      axis, detrend, scaling, func=modified_dft)

    # obtain the positive freqs.
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    
    # num. segments that fit into pro samples of len nfft with % overlap
    nsegs = int((data.shape[axis] - nfft) // (nfft * (1-overlap)) + 1)
    shape = list(data.shape)
    shape[axis] = nsegs

    # compute the segment times
    if boundary:
        time = 1 / fs * np.arange(0, data.shape[axis] - nfft + 1, nfft-noverlap)
    else:
        time = 1 / fs * np.arange(nfft//2, data.shape[axis] + 1 - nfft//2, 
                                  nfft-noverlap)

    # return producer from welch gen func with each yielded 
    result = producer(genfunc, chunksize=len(freqs), axis=axis, shape=shape)
    return freqs, time, result

