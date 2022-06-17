"""
Module description and usage
"""

import numpy as np

from scipy import signal as sps

from openseize.core.producer import producer
from openseize.core.arraytools import pad_along_axis
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

    # build a kaiser antialiasing filter and fetch its coeffs.
    fstop = kwargs.pop('fstop', fs // M)
    fpass = kwargs.pop('fpass', fstop - fstop // 10)
    gpass, gstop = kwargs.pop('gpass', 1), kwargs.pop('gstop', 40)
    h = Kaiser(fpass, fstop, fs, gpass, gstop).coeffs

    # data subsequences are left shifted upto M-1 so prepad
    y = pad_along_axis(arr, [M-1, 0], axis=axis)

    # build filters polyphase components
    p_len = int(np.ceil(len(h) / M))
    # ensure all components will have p_len
    p = pad_along_axis(h, [0, M * p_len - len(h)])
    p = p.reshape((M, p_len), order='F')

    # build data subsequences
    u_len = int(np.ceil(y.shape[axis]  / M))
    # ensure all subsequences will have u_len
    u = pad_along_axis(y, [0, M * u_len - y.shape[axis]], axis=axis)
    
    # channels on 0th axis and reshape to (chs, M, u_len)
    u = u.T if axis == 0 else u
    u = u.reshape((u.shape[0], M, u_len), order='F')
    u = np.flip(u, axis=1)

    # Each p is very small so it takes longer to use an FFT method
    result = np.zeros((y.shape[0], (u_len + p_len - 1)))
    for idx in range(y.shape[0]):
        for m in range(M):
            result[idx] += np.convolve(p[m], u[idx, m])
    
    # no need to return full convolution in a downsample op

    return result


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

    for idx, arr in enumerate(pro):
        
        # Build data subsequences u_m
        if idx < 1:
            # subsequences are left shifted upto M-1 so prepad
            arr = pad_along_axis(arr, [M-1, 0], axis=axis)
        # FIXME all other arrays will need to be padded to but with last
        # values?

        u_len = int(np.ceil(arr.shape[axis] / M))
        # ensure all subsequences will have u_len
        u = pad_along_axis(arr, [0, M * u_len - arr.shape[axis]], axis=axis)

        # channels on 0th axis and reshape to (chs, M, u_len)
        u = u.T if axis == 0 else u
        u = u.reshape((u.shape[0], M, u_len), order='F')
        u = np.flip(u, axis=1)

        last_arr




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time

    from openseize.io import edf

    edf_path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'

    def data(edf_path, start, stop, channels=None):
        """Returns an array of data from an edf file."""

        reader = edf.Reader(edf_path)
        return reader.read(start, stop, channels=channels)


    arr = data(edf_path, 0, 30000000)

    # polyphase decomposition
    t0 = time.perf_counter()
    d = downsample(arr, fs=5000, M=10)
    print('Polyphase time = {}'.format(time.perf_counter() - t0))

    # Standard method 
    t0 = time.perf_counter()
    filt = Kaiser(fpass=450, fstop=500, fs=5000)
    ground_truth = []
    for signal in arr:
        x = np.convolve(signal, filt.coeffs, mode='full')
        ground_truth.append(x[::10])
    ground_truth = np.array(ground_truth)
    print('Filter-decimate time = {}'.format(time.perf_counter() - t0))

    fig, axarr = plt.subplots(arr.shape[0], 1)
    for idx, (a, b) in enumerate(zip(d, ground_truth)):
        axarr[idx].plot(a[100000:200000], color='tab:blue')
        axarr[idx].plot(b[100000:200000], color='tab:orange', alpha=0.5)
    plt.show()

