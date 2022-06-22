"""
Module description and usage
"""

import numpy as np

from scipy import signal as sps

from openseize.core.producer import producer
from openseize.core.arraytools import pad_along_axis, slice_along_axis
from openseize.core.numerical import convolve_slicer
from openseize.filtering.fir import Kaiser


def _downsample(arr, fs, M, axis=-1, **kwargs):
    """Polyphase decomposition downsampling of an array by an integer
    factor.

    Args:
        arr: 2D numpy array
            An array of datapoints to be downsampled along axis.
        fs: int
            Sampling frequency of data in arr
        M: int
            Integer downsampling factor
        axis: int
            Axis along which antialiasing filter and decimation will occur.
            Default is last axis.
        kwargs:
            Any valid arguments for a Kaiser lowpass filter. If none 
            provided the defaults for this filter are:

                fstop = fs // M
                fpass = fstop - fstop // 10
                gpass = 1
                gstop = 40 dB
        
    Returns: An array of antialiased and decimated signals.
    """

    print('arr shape = ', arr.shape)

    # build a kaiser antialiasing filter and fetch its coeffs.
    fstop = kwargs.pop('fstop', fs // M)
    fpass = kwargs.pop('fpass', fstop - fstop // 10)
    gpass, gstop = kwargs.pop('gpass', 1), kwargs.pop('gstop', 40)
    h = Kaiser(fpass, fstop, fs, gpass, gstop).coeffs

    print('filter length = ', len(h))

    # data subsequences are left shifted upto M-1 so prepad
    y = pad_along_axis(arr, [M-1, 0], axis=axis)

    print('padded signals shape = ', y.shape)

    # build filters polyphase components
    p_len = int(np.ceil(len(h) / M))
    # ensure all components will have p_len
    p = pad_along_axis(h, [0, M * p_len - len(h)])
    p = p.reshape((M, p_len), order='F')

    print('polyphase components shape = ', p.shape)

    # build data subsequences
    u_len = int(np.ceil(y.shape[axis]  / M))
    print('u_len = ', u_len)
    
    # ensure all subsequences will have u_len
    u = pad_along_axis(y, [0, M * u_len - y.shape[axis]], axis=axis)
    
    # channels on 0th axis and reshape to (chs, M, u_len)
    u = u.T if axis == 0 else u
    u = u.reshape((u.shape[0], M, u_len), order='F')
    u = np.flip(u, axis=1)

    print('subsequences shape = ', u.shape)

    # Each p is very small so it takes longer to use an FFT method
    result = np.zeros((y.shape[0], u_len + p_len - 1))
    for idx in range(y.shape[0]):
        for m in range(M):
            result[idx] += np.convolve(p[m], u[idx, m])

    print('result shape = ', result.shape)

    # upfirdn matches result but I need to figure out how to slice it

    # no need to return full convolution in a downsample op
    a = (M - 1 + (len(h)-1)//2) // M
    b = M * u_len - y.shape[axis] + arr.shape[axis] // (
        M + bool(arr.shape[axis] % M))
    result = slice_along_axis(result, start=a, stop=b)

    return result 


def _downsample2(data, M, fs, chunksize, axis=-1, **kwargs):
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
        An array or producer of arrays with datatype matching data's 
        datatype.

    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 12 "Multirate Signal Processing"
    """

    pro = producer(data, chunksize, axis)

    # build and get coeffs of the kaiser antialiasing filter
    fstop = kwargs.pop('fstop', fs // M)
    fpass = kwargs.pop('fpass', fstop - fstop // 10)
    gpass, gstop = kwargs.pop('gpass', 1), kwargs.pop('gstop', 40)
    h = Kaiser(fpass, fstop, fs, gpass, gstop).coeffs

    # build M polyphase components padding upto max comp. length
    p_len = int(np.ceil(len(h) / M))
    p = pad_along_axis(h, [0, M * p_len - len(h)])
    p = p.reshape((M, p_len), order='F')

    # build cache holding M-1 samples
    cshape = list(pro.shape)
    cshape[axis] =  M-1
    cache = np.zeros(cshape)

    for idx, arr in enumerate(pro):
        
        # Build data subsequences u_m
        # data subsequences need to reach back M-1 samples so prepad
        arr = np.concatenate((cache, arr), axis=axis)
        
        u_len = int(np.ceil(arr.shape[axis] / M))
        # ensure all subsequences will have u_len
        u = pad_along_axis(arr, [0, M * u_len - arr.shape[axis]], axis=axis)

        # channels on 0th axis and reshape to (chs, M, u_len)
        u = u.T if axis == 0 else u
        u = u.reshape((u.shape[0], M, u_len), order='F')
        u = np.flip(u, axis=1)

        # Each p is very small so it takes longer to use an FFT method
        result = np.zeros((arr.shape[0], (u_len + p_len - 1)))
        for sig_idx in range(arr.shape[0]):
            for m in range(M):
                result[sig_idx] += np.convolve(p[m], u[sig_idx, m])
        
        if idx > 0:
            result = convolve_slicer(result, p.shape, u.shape, mode='same',
                    axis=axis)
            
        
        cache = slice_along_axis(arr, -(M-1), None, axis=axis)

        yield result

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
        An array or producer of arrays with datatype matching data's 
        datatype.

    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 12 "Multirate Signal Processing"
    """

    x = producer(data, chunksize, axis)
    y = producer(data, chunksize, axis)
    z = producer(data, chunksize, axis)

    # build and get coeffs of the kaiser antialiasing filter
    fstop = kwargs.pop('fstop', fs // M)
    fpass = kwargs.pop('fpass', fstop - fstop // 10)
    gpass, gstop = kwargs.pop('gpass', 1), kwargs.pop('gstop', 40)
    h = Kaiser(fpass, fstop, fs, gpass, gstop).coeffs

    # additional samples to avoid boundary effects
    q = int(np.ceil(len(h) / M) * M)
    q += bool(q % M)
    print(q)
    boundary_shape = list(x.shape)
    boundary_shape[axis] = q
 
    ix, iy, iz = (iter(pro) for pro in [x,y,z])
    #
    pre_bound = np.zeros(boundary_shape)
    #
    next(iz)
    post_bound = slice_along_axis(next(iz), 0, q)
    #
    padded = np.concatenate((pre_bound, next(iy), post_bound), axis=axis)
    ds = sps.resample_poly(padded, up=1, down=M, axis=axis, window=h)
    result = slice_along_axis(ds, start=q//M, stop=-q//M, axis=axis)
    yield result

    nchunks = y.shape[axis] // y.chunksize + bool(y.shape[axis]
            % y.chunksize)
    print(nchunks)
    for idx, (last_arr, arr, next_arr) in enumerate(zip(ix, iy, iz),1):

            pre_bound = slice_along_axis(last_arr, -q, axis=axis)
            
            #if next_arr.shape[axis] < z.chunksize:
            if idx == nchunks - 2: # TODO UNDERSTAND
                # we are just before the last chunk
                arr = np.concatenate((arr, next_arr), axis=axis)
                padded = np.concatenate((pre_bound, arr), axis=-1)
                ds = sps.resample_poly(padded, up=1, down=M, axis=axis, 
                                       window=h)
                yield slice_along_axis(ds, start=q//M, stop=None,  axis=axis)
            
            else:

                post_bound = slice_along_axis(next_arr, 0, q, axis=axis)
                padded = np.concatenate((pre_bound, arr, post_bound), axis=axis)
                ds = sps.resample_poly(padded, up=1, down=M, axis=axis, 
                                       window=h)

                yield slice_along_axis(ds, start=q//M, stop=-q//M,  axis=axis)


    """
    for idx, (last_arr, arr) in enumerate(zip(ix, iy), 1):

        try:
            print('idx : ', idx)
            next_arr = next(iz)
            print ('arr shape = ', arr.shape)
            print ('next arr shape = ', next_arr.shape)
            pre_bound = slice_along_axis(last_arr, -q, axis=axis)
            post_bound = slice_along_axis(next_arr, 0, q, axis=axis)
            padded = np.concatenate((pre_bound, arr, post_bound), axis=axis)

            ds = sps.resample_poly(padded, up=1, down=M, axis=axis, window=h)
            print('ds shape = ', ds.shape)
            result = slice_along_axis(ds, start=q//M, stop=-q//M,  axis=axis)
            print('result shape = {}'.format(result.shape))
            yield result
        
        except StopIteration:
            print('last arrays')
            print(last_arr.shape, arr.shape)
            pre_bound  = slice_along_axis(last_arr, -q, axis=axis)
            padded = np.concatenate((pre_bound, arr), axis=axis)
            print('padded shape', padded.shape)
            ds = sps.resample_poly(padded, up=1, down=M, axis=axis, window=h)
            print('ds shape = ', ds.shape)
            result = slice_along_axis(ds, start=q//M, stop=None,  axis=axis)
            print(result.shape)
            yield result
    """




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time

    from openseize.io import edf

    edf_path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'

    def data(edf_path, start, stop, channels=None):
        """Returns an array of data from an edf file."""

        reader = edf.Reader(edf_path)
        return reader.read(start, stop, channels=channels)


    arr = data(edf_path, 0, 3000000)

    """
    # polyphase decomposition
    t0 = time.perf_counter()
    pp = _downsample(arr, fs=5000, M=10)
    print('Polyphase time = {}'.format(time.perf_counter() - t0))
    """
    

    
    t0 = time.perf_counter()
    res = downsample(arr, M=10, fs=5000, chunksize=100000, axis=-1)
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
        axarr[idx].plot(a[-50000:], color='tab:blue', label='pp')
        #axarr[idx].plot(b[0:50000], color='tab:green', alpha=0.5,
        #label='standard')
        axarr[idx].plot(c[-50000:], color='tab:red', alpha=0.5,
        label='scipy')
    plt.legend()
    plt.ion()
    plt.show()

