"""Tools for downsampling, upsampling and resampling data using polyphase
decomposition methods (see scipy resample_polyphase).

This module contains the following functions:

    downsample:
        A function that downsamples an array or producer of arrays by an
        integer decimation factor in conjuction with an antialiasing filter.
        
        Typical usage example:
        pro = downsample(data, M=10, fs=5000, chunksize=10000, axis=-1)
        #returns a producer or ndarray of M-downsampled data where the
        downsampling axis is the last axis.

    upsample:
    resample:
"""

import numpy as np
from scipy import signal as sps

from openseize.core.producer import producer
from openseize.core.arraytools import pad_along_axis, slice_along_axis
from openseize.core.numerical import convolve_slicer
from openseize.filtering.fir import Kaiser


def downsample(data, M, fs, chunksize, axis=-1, **kwargs):
    """Downsamples an array or producer of arrays using polyphase 
    decomposition by an integer factor decimation factor.

    Args:
        data: ndarray or producer of ndarrays
            The data to be downsampled.
        M: int
            The decimation factor describing which Mth samples of data
            survive decimation. (E.g. M=10 -> every 10th sample survives)
        fs: int
            The sampling rate of data in Hz.
        chunksize: int
            The number of samples to hold in memory during downsampling.
            This method will require ~ 3 times chunksize in memory.
        axis: int
            The axis of data along which downsampling will occur.
        kwargs:
            Any valid keyword for a Kaiser lowpass filter. The default 
            values for this antialiasing filter are:

                fstop: int
                    The stop band edge frequency. Defaults to fs // M.
                fpass: int
                    The pass band edge frequency. Must be less than fstop.
                    Defaults to fstop - fstop // 10.
                gpass: int
                    The pass band attenuation in dB. Defaults to a max loss
                    in the passband of 1 dB ~ 11% amplitude loss.
                gstop: int
                    The max attenuation in the stop band in dB. Defaults to
                    40 dB or 99%  amplitude attenuation.

    Returns:
        A producer of arrays.

    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 12 "Multirate Signal Processing"
        2. Polyphase implementation: scipy.signal.resample_poly
    """

    # iterators yielding prior, current and next chunks of data
    x = producer(data, chunksize, axis)
    y = producer(data, chunksize, axis)
    z = producer(data, chunksize, axis)
    iprior, icurrent, inext = (iter(pro) for pro in [x,y,z])

    # kaiser antialiasing filter coeffecients
    fstop = kwargs.pop('fstop', fs // M)
    fpass = kwargs.pop('fpass', fstop - fstop // 10)
    gpass, gstop = kwargs.pop('gpass', 1), kwargs.pop('gstop', 40)
    h = Kaiser(fpass, fstop, fs, gpass, gstop).coeffs

    # num. boundary pts to cover left & right convolve overhangs
    # must be divisible by M for integer slicing after downsampling
    overhang = int(np.ceil(len(h) / M) * M) 
    
    # initialize the first left boundary and right boundaries
    left_bound_shape = list(data.shape)
    left_bound_shape[axis] = overhang
    left_bound = np.zeros(left_bound_shape)
    next(inext)
    right_bound = slice_along_axis(next(inext), 0, overhang,  axis=axis) 
    
    # yield initial boundary corrected downsampled chunk
    current = next(icurrent)
    padded = np.concatenate((left_bound, current, right_bound), axis=axis)
    downed = sps.resample_poly(padded, up=1, down=M, axis=axis, window=h)
    yield slice_along_axis(downed, overhang//M, -overhang//M, axis=axis)

    # downsample remaining cnt chunks
    cnt = z.shape[axis] // chunksize + bool(z.shape[axis] % chunksize) - 1
    for n, (last, curr, nxt) in enumerate(zip(iprior, icurrent, inext), 1):
        
        left_bound = slice_along_axis(last, -overhang, axis=axis)
     
        if n < cnt - 1:
            right_bound = slice_along_axis(nxt, 0, overhang, axis=axis)
        else:
            # at cnt-1 chunks concantenate next to current
            curr = np.concatenate((curr, nxt), axis=axis) 
            right_bound = np.zeros(left_bound.shape)

        padded = np.concatenate((left_bound, curr, right_bound), axis)
        downed = sps.resample_poly(padded, 1, M, axis=axis, window=h)
        yield slice_along_axis(downed, overhang//M, -overhang//M, axis=axis)


def upsample(data, L, fs, chunksize, axis=-1, **kwargs)
    """Upsamples an array or producer of arrays using polyphase 
    decomposition by an integer expansion factor.

    Args:
        data: ndarray or producer of ndarrays
            The data to be upsampled.
        L: int
            The expansion factor. L-1 interpolated values will be inserted
            between consecutive samples in data along axis.
        fs: int
            The sampling rate of data in Hz.
        chunksize: int
            The number of samples to hold in memory during upsampling.
            This method will require ~ 3 times chunksize in memory.
        axis: int
            The axis of data along which downsampling will occur.
        kwargs:
            Any valid keyword for a Kaiser lowpass filter. The default 
            values for this antialiasing filter are:

                fstop: int
                    The stop band edge frequency. Defaults to fs // M.
                fpass: int
                    The pass band edge frequency. Must be less than fstop.
                    Defaults to fstop - fstop // 10.
                gpass: int
                    The pass band attenuation in dB. Defaults to a max loss
                    in the passband of 1 dB ~ 11% amplitude loss.
                gstop: int
                    The max attenuation in the stop band in dB. Defaults to
                    40 dB or 99%  amplitude attenuation.

    Returns:
        A producer of arrays.

    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 12 "Multirate Signal Processing"
        2. Polyphase implementation: scipy.signal.resample_poly
    """

    # iterators yielding prior, current and next chunks of data
    x = producer(data, chunksize, axis)
    y = producer(data, chunksize, axis)
    z = producer(data, chunksize, axis)
    iprior, icurrent, inext = (iter(pro) for pro in [x,y,z])

    # kaiser antialiasing filter coeffecients
    fstop = kwargs.pop('fstop', fs // M)
    fpass = kwargs.pop('fpass', fstop - fstop // 10)
    gpass, gstop = kwargs.pop('gpass', 1), kwargs.pop('gstop', 40)
    h = Kaiser(fpass, fstop, fs, gpass, gstop).coeffs

    # num. boundary pts to cover left & right convolve overhangs
    overhang = len(h) - 1

    # initialize the first left boundary and right boundaries
    left_bound_shape = list(data.shape)
    left_bound_shape[axis] = overhang
    left_bound = np.zeros(left_bound_shape)
    next(inext)
    right_bound = slice_along_axis(next(inext), 0, overhang,  axis=axis) 

    # yield initial boundary corrected upsampled chunk
    current = next(icurrent)
    padded = np.concatenate((left_bound, current, right_bound), axis=axis)
    downed = sps.resample_poly(padded, up=L, down=1, axis=axis, window=h)
    yield slice_along_axis(downed, overhang * L, -overhang * L, axis=axis)

    # upsample remaining cnt chunks
    cnt = z.shape[axis] // chunksize + bool(z.shape[axis] % chunksize) - 1
    for n, (last, curr, nxt) in enumerate(zip(iprior, icurrent, inext), 1):
        
        left_bound = slice_along_axis(last, -overhang, axis=axis)
     
        if n < cnt - 1:
            right_bound = slice_along_axis(nxt, 0, overhang, axis=axis)
        else:
            # at cnt-1 chunks concantenate next to current
            curr = np.concatenate((curr, nxt), axis=axis) 
            right_bound = np.zeros(left_bound.shape)

        padded = np.concatenate((left_bound, curr, right_bound), axis)
        d = sps.resample_poly(padded, L, 1, axis=axis, window=h)
        yield slice_along_axis(d, overhang * L, -overhang * L, axis=axis)




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time

    from openseize.io import edf

    edf_path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'

    def data(edf_path, start, stop, channels=None):
        """Returns an array of data from an edf file."""

        reader = edf.Reader(edf_path)
        return reader.read(start, stop, channels=channels)


    arr = data(edf_path, 0, 2999997)

    
    t0 = time.perf_counter()
    res = downsample(arr, M=10, fs=5000, chunksize=10000, axis=-1)
    pp = np.concatenate([x for x in res], axis=-1)
    print('Polyphase time = {}'.format(time.perf_counter() - t0))

    """
    # Standard method 
    t0 = time.perf_counter()
    filt = Kaiser(fpass=450, fstop=500, fs=5000)
    x = []
    for signal in arr:
        a = np.convolve(signal, filt.coeffs, 'full')
        x.append(a[::10])
    x = np.array(x)
    print('x shape = ', x.shape)
    print('Filter-decimate time = {}'.format(time.perf_counter() - t0))
    """

    #scipy resample_poly
    t0 = time.perf_counter()
    filt = Kaiser(fpass=450, fstop=500, fs=5000)
    y = sps.resample_poly(arr, 1, 10, axis=-1, window=filt.coeffs)
    #y = sps.upfirdn(filt.coeffs, arr, up=1, down=10, axis=-1)
    print('scipy polyphase time = {}'.format(time.perf_counter() - t0))

    fig, axarr = plt.subplots(arr.shape[0], 1, figsize=(4,8))
    for idx, (a, c) in enumerate(zip(pp, y)):
        axarr[idx].plot(a[:], color='tab:blue', label='pp')
        #axarr[idx].plot(b[0:50000], color='tab:green', alpha=0.5,
        #label='standard')
        axarr[idx].plot(c[:], color='tab:red', alpha=0.5,
        label='scipy')
    plt.legend()
    plt.ion()
    plt.show()

