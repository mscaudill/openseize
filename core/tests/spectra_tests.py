"""A module for testing openseize's spectral functions.

Typical usage example:
    !pytest spectra_tests.py::<TEST NAME>
"""

import pytest
import numpy as np
import scipy.signal as sps

from openseize import producer
from openseize.core.numerical import periodogram, welch
from openseize.filtering.fir import Kaiser

def test_periodogram_arrs():
    """Test if the openseize periodogram function matches scipy's peridogram
    for a arrays whose sample axis length varies."""

    rng = np.random.default_rng(1234)
    
    lengths = rng.integers(10000, high=77000, size=50)
    arrs = [rng.random((2, 3, l)) for l in lengths]

    # periodogram parameters
    fs = 1000
    nfft = None
    window='hann'
    detrend = 'constant'
    scaling = 'density'
    return_onesided=True
    axis = -1

    for arr in arrs:

        # openseize result
        op_f, op_res = periodogram(arr, fs, nfft, window, axis, detrend, 
                                   scaling) 

        # scipy result
        sp_f, sp_res = sps.periodogram(arr, fs, window, nfft, detrend,
                                       return_onesided, scaling, axis)

        assert np.allclose(op_f, sp_f)
        assert np.allclose(op_res, sp_res)


def test_periodogram_nfft():
    """Test if openseize periodogram result matches scipy result for various
    numbers of FFT points."""

    rng = np.random.default_rng(1234)
    arr = rng.random((3, 4, 132023))

    nffts = list(rng.integers(low=500, high=132023, size=50))
    nffts.append(None) # None option sets nfft == arr.shape[-1]

    # periodogram parameters
    fs = 6000
    window='hann'
    detrend = 'constant'
    scaling = 'density'
    return_onesided=True
    axis = -1

    for nfft in nffts:

        # openseize result
        op_f, op_res = periodogram(arr, fs, nfft, window, axis, detrend, 
                                   scaling) 

        # scipy result
        sp_f, sp_res = sps.periodogram(arr, fs, window, nfft, detrend,
                                       return_onesided, scaling, axis)

        assert np.allclose(op_f, sp_f)
        assert np.allclose(op_res, sp_res)


def test_periodogram_windows():
    """Test if openseize periodogram result matches scipy result for various
    scipy signal windows."""

    windows = 'hann hamming boxcar nuttall blackman bartlett cosine'.split()

    rng = np.random.default_rng(1234)
    arr = rng.random((3, 4, 132026))

    # periodogram parameters
    fs = 1000
    nfft = None
    detrend = 'constant'
    scaling = 'density'
    return_onesided=True
    axis = -1

    for win in windows:

        # openseize result
        op_f, op_res = periodogram(arr, fs, nfft, win, axis, detrend, 
                                   scaling) 

        # scipy result
        sp_f, sp_res = sps.periodogram(arr, fs, win, nfft, detrend,
                                       return_onesided, scaling, axis)

        assert np.allclose(op_f, sp_f)
        assert np.allclose(op_res, sp_res)


def test_periodogram_scaling():
    """Test if openseize periodogram result matches scipy result for
    different scaling options."""

    scales = ['density', 'spectrum']

    rng = np.random.default_rng(1234)
    arr = rng.random((3, 4, 132026))

    # periodogram parameters
    fs = 1000
    nfft = None
    window='hann'
    detrend = 'constant'
    return_onesided=True
    axis = -1

    for scale in scales:

        # openseize result
        op_f, op_res = periodogram(arr, fs, nfft, window, axis, detrend, 
                                   scale) 

        # scipy result
        sp_f, sp_res = sps.periodogram(arr, fs, window, nfft, detrend,
                                       return_onesided, scale, axis)

        assert np.allclose(op_f, sp_f)
        assert np.allclose(op_res, sp_res)


def test_periodogram_arr_error():
    """Test that empty array error raises ValueError."""

    arr = np.ones((4,0))

    # periodogram parameters
    fs = 1000
    nfft = None
    window='hann'
    detrend = 'constant'
    scaling = 'density'
    return_onesided=True
    axis = -1

    # test that ValueError is raised by Numpy fft
    with pytest.raises(ValueError):

        op_f, op_res = periodogram(arr, fs, nfft, window, axis, detrend, 
                                   scaling) 
    

def test_welch_pros():
    """Test if Openseize's Welch result matches scipy's result for a variety
    of producer sizes."""


    rng = np.random.default_rng(1234)
    
    lengths = rng.integers(10000, high=77000, size=50)
    arrs = [rng.random((2, 3, l)) for l in lengths]

    # welch parameters
    fs = 1000
    nfft = 1000
    window='hann'
    overlap = 0.6
    detrend = 'constant'
    scaling = 'density'
    return_onesided=True
    axis = -1

    for arr in arrs:

        pro = producer(arr, chunksize=1000, axis=-1)
        # openseize result
        op_f, op_segs = welch(pro, fs, nfft, window, overlap, axis, detrend, 
                             scaling)
        
        # get average of all periodograms in op_segs
        op_res = 0
        for cnt, x in enumerate(op_segs, 1):

            op_res = op_res + 1 / cnt * (x - op_res)

    
        # scipy result
        sp_f, sp_res = sps.welch(arr, fs=fs, window=window, nperseg=nfft,
                noverlap=int(overlap*nfft), detrend=detrend,
                return_onesided=return_onesided, scaling=scaling, axis=axis)

        assert np.allclose(op_f, sp_f)
        assert np.allclose(op_res, sp_res)


def test_welch_overlaps():
    """Test if Openseize's Welch result matches scipy's result for a variety
    of overlap amounts."""


    rng = np.random.default_rng(1234)
    arr = rng.random((3, 4, 762065))
    
    # vary overlaps between 0.1 and 0.9
    overlaps = np.arange(0.1, 1, 0.1)

    # welch parameters
    fs = 5000
    nfft = 10000
    window='hann'
    detrend = 'constant'
    scaling = 'density'
    return_onesided=True
    axis = -1

    for overlap in overlaps:

        pro = producer(arr, chunksize=1000, axis=-1)
        # openseize result
        op_f, op_segs = welch(pro, fs, nfft, window, overlap, axis, detrend, 
                             scaling)
        
        # get average of all periodograms in op_segs
        op_res = 0
        for cnt, x in enumerate(op_segs, 1):

            op_res = op_res + 1 / cnt * (x - op_res)

    
        # scipy result
        sp_f, sp_res = sps.welch(arr, fs=fs, window=window, nperseg=nfft,
                noverlap=int(overlap*nfft), detrend=detrend,
                return_onesided=return_onesided, scaling=scaling, axis=axis)

        assert np.allclose(op_f, sp_f)
        assert np.allclose(op_res, sp_res)


def test_welch_windows():
    """Test if openseize welch result matches scipy result for various
    scipy signal windows."""

    windows = 'hann hamming boxcar nuttall blackman bartlett cosine'.split()

    rng = np.random.default_rng(1234)
    arr = rng.random((3, 4, 132026))

    # periodogram parameters
    fs = 1000
    nfft = 1000
    overlap=0.3
    detrend = 'constant'
    scaling = 'density'
    return_onesided=True
    axis = -1

    for window in windows:

        pro = producer(arr, chunksize=1000, axis=-1)
        # openseize result
        op_f, op_segs = welch(pro, fs, nfft, window, overlap, axis, detrend, 
                             scaling)
        
        # get average of all periodograms in op_segs
        op_res = 0
        for cnt, x in enumerate(op_segs, 1):

            op_res = op_res + 1 / cnt * (x - op_res)

    
        # scipy result
        sp_f, sp_res = sps.welch(arr, fs=fs, window=window, nperseg=nfft,
                noverlap=int(overlap*nfft), detrend=detrend,
                return_onesided=return_onesided, scaling=scaling, axis=axis)

        assert np.allclose(op_f, sp_f)
        assert np.allclose(op_res, sp_res)


def test_welch_scaling():
    """Test if openseize welch result matches scipy result for various
    scipy ."""

    
    scales = ['density','spectrum']

    rng = np.random.default_rng(1234)
    arr = rng.random((3, 4, 132026))

    # periodogram parameters
    fs = 1000
    nfft = 1000
    overlap=0.2
    window='boxcar'
    detrend = 'constant'
    return_onesided=True
    axis = -1

    for scaling in scales:

        pro = producer(arr, chunksize=1000, axis=-1)
        # openseize result
        op_f, op_segs = welch(pro, fs, nfft, window, overlap, axis, detrend, 
                             scaling)
        
        # get average of all periodograms in op_segs
        op_res = 0
        for cnt, x in enumerate(op_segs, 1):

            op_res = op_res + 1 / cnt * (x - op_res)

    
        # scipy result
        sp_f, sp_res = sps.welch(arr, fs=fs, window=window, nperseg=nfft,
                noverlap=int(overlap*nfft), detrend=detrend,
                return_onesided=return_onesided, scaling=scaling, axis=axis)

        assert np.allclose(op_f, sp_f)
        assert np.allclose(op_res, sp_res)


def test_welch_nfft():
    """Test if openseize welch result matches scipy result for various
    numbers of FFT points."""

    rng = np.random.default_rng(1234)
    arr = rng.random((3, 4, 132023))

    nffts = list(rng.integers(low=1000, high=132023, size=50))

    # periodogram parameters
    fs = 6000
    window='hann'
    overlap=0.5
    detrend = 'constant'
    scaling = 'density'
    return_onesided=True
    axis = -1

    for nfft in nffts:

        pro = producer(arr, chunksize=1000, axis=-1)
        # openseize result
        op_f, op_segs = welch(pro, fs, nfft, window, overlap, axis, detrend, 
                             scaling)
        
        # get average of all periodograms in op_segs
        op_res = 0
        for cnt, x in enumerate(op_segs, 1):

            op_res = op_res + 1 / cnt * (x - op_res)

    
        # scipy result
        sp_f, sp_res = sps.welch(arr, fs=fs, window=window, nperseg=nfft,
                noverlap=int(overlap*nfft), detrend=detrend,
                return_onesided=return_onesided, scaling=scaling, axis=axis)

        assert np.allclose(op_f, sp_f)
        assert np.allclose(op_res, sp_res)


