# open data
# write downsampler
# build filter and get coeffs
# run downsampler on data
# compare results with direct filtering and decimation

import numpy as np
import scipy as sp

from openseize.io import edf
from openseize.filtering.fir import Kaiser
from openseize.core import arraytools

def data(edf_path, start, stop, channels=None):
    """Returns an array of data from an edf file."""

    reader = edf.Reader(edf_path)
    return reader.read(start, stop, channels=channels)

def kaiser(fpass, fstop, fs, gpass=1.0, gstop=40, **kwargs):
    """Designs a Kaiser filter."""

    filt = Kaiser(fpass, fstop, fs, gpass, gstop, **kwargs)
    return filt

def _downsample(arr, M, fs):
    """Ver. 1 Polyphase M-downsampling of arr."""

    fpass = fs / M
    fstop = fpass + 100
    gpass = 1
    gstop = 40
    
    filt = kaiser(fpass, fstop, fs=fs, gpass=gpass, gstop=gstop)
    h = filt.coeffs

    y = arraytools.pad_along_axis(arr, [M-1,0], axis=0)

    res = []

    max_p_length = int(np.ceil(len(h) / M))
    max_u_length = int(np.ceil((len(arr) + M-1) / M)) # understand this
    #print('len(p) = ', max_p_length)
    #print('len(u) = ', max_u_length)
    for m in range(M):

        pm = h[m::M]
        pm = arraytools.pad_along_axis(pm, [0, max_p_length - len(pm)], axis=0)

        um = y[M-m-1::M]
        um = arraytools.pad_along_axis(um, [0, max_u_length - len(um)],
                axis=0)
        print(um.shape)
        
        y_m = np.convolve(pm, um, mode='full')
        #print(y_m.shape)
        res.append(y_m)
    return res

def downsample(arr, fs, M, axis=-1):
    """Ver. 2 Avoids loops using reshape on p's and u's """

    fpass = fs // M
    fstop = fpass + 100
    gpass, gstop = 1, 40

    filt = kaiser(fpass, fstop, fs=fs, gpass=gpass, gstop=gstop)
    h = filt.coeffs

    # u_m are advanced in time from m=0 upto M-1, so pad
    padded = arraytools.pad_along_axis(arr, [M-1,0], axis=axis)

    # polyphase components of h
    p_len = int(np.ceil(len(h) / M))
    p = arraytools.pad_along_axis(h, [0, M * p_len - len(h)], axis=0)
    p = p.reshape((M, p_len), order='F') 

    # shifted decimations
    u_len = int(np.ceil(padded.shape[axis] / M)) 
    u = arraytools.pad_along_axis(padded, [0, M * u_len
                                  - padded.shape[axis]], axis=axis)
    u = u.reshape((M, u_len), order='F')
    u = np.flipud(u)

    result = np.zeros((1, (u_len + p_len - 1)))
    for m in range(M):
        result += np.convolve(p[m], u[m])

    # slower due to small size of each p
    #result = sp.signal.oaconvolve(p, u, axes=-1)
    return result

def ndownsample(arr, fs, M axis=-1):
    """Ver. 3 Downsamples multichannel data."""


    














if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time

    edf_path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'

    arr = data(edf_path, 0, 80000000)
    signal = arr[0]

    
    t0 = time.perf_counter()
    d = downsample(signal, M=10, fs=5000)
    print('Polyphase time = {}'.format(time.perf_counter() - t0))

    t0 = time.perf_counter()
    filt = kaiser(fpass=500, fstop=100, fs=5000)
    ground_truth = np.convolve(signal, filt.coeffs, mode='full')
    ground_truth = ground_truth[::10]
    print('Filter-decimate time = {}'.format(time.perf_counter() - t0))

    """
    plt.plot(d)
    plt.plot(ground_truth)
    plt.show()
    """
    
