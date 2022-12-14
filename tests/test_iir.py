"""A module for testing filtering tools and their subfunctions.

Typical usage example:
    !pytest iir_tests.py::<TEST_NAME>
"""

import pytest
import numpy as np
import scipy.signal as sps

from openseize import producer
from openseize.core import numerical as nm
from openseize.filtering import iir


def build_filter(cls, *args, **kwargs):
    """Builds an IIR filter passing in all args and kwargs."""

    return cls(*args, **kwargs)


def test_sos_coeffs0():
    """Determines if openseizes SOS coeffecients for a filter matches scipys
    iirdesign for a variety of iir filters."""

    # build an openseize filter & fetch coeffs
    fpass = np.array([400,800])
    fstop = np.array([300, 900])
    gpass = 2
    gstop = 40
    fs = 5000

    types = iir.Butter, iir.Cheby1, iir.Cheby2, iir.Ellip

    for ftype in types:
       
        # build openseize filter and get its sos coeffs
        filt = build_filter(ftype, fpass, fstop, gpass=gpass, gstop=gstop,
                            fs=fs, fmt='sos')

        o_sos = filt.coeffs

        # build scipy filter and get its sos coeffs
        sp_sos = sps.iirdesign(2*fpass/fs, 2*fstop/fs, gpass=gpass, gstop=gstop,
                               ftype=ftype.__name__.lower(),
                               output='sos')

        assert np.allclose(o_sos, sp_sos)


def test_sos_coeffs1():
    """Test if openseize's sos coeffs match scipys for a variety of pass and
    stop bands."""

    # lowpass, bandpass highpass
    fpasses = [300, np.array([200, 450]), 1000]
    fstops = [400, np.array([100, 600]), 800]

    gpass, gstop = 1, 30
    fs = 5000

    for fpass, fstop in zip(fpasses, fstops):
        
        # build openseize filter and get its sos coeffs
        filt = build_filter(iir.Butter, fpass, fstop, gpass=gpass, 
                            gstop=gstop,fs=fs, fmt='sos')
        
        o_sos = filt.coeffs

        # build scipy filter and get its sos coeffs
        sp_sos = sps.iirdesign(2*fpass/fs, 2*fstop/fs, gpass=gpass, 
                               gstop=gstop, ftype='butter', output='sos')

        assert np.allclose(o_sos, sp_sos)


def test_sosfilt_pros():
    """Test if openseize's sosfilt matches scipy's sosfilt restults for
    a Butterworth IIR using sos fmt for a variety of producer sizes."""

    rng = np.random.default_rng(0)
    axis = 1
    fs = 500

    lengths = rng.integers(10000, 230000, size=10) 

    for length in lengths:
        
        # build array and producer
        arr = rng.random((3, length, 6))
        pro = producer(arr, chunksize=10000, axis=axis)

        # build an openseize filter and call it
        filt = build_filter(iir.Butter, fpass=100, fstop=200, fs=fs)
        pro_filt = filt(pro, chunksize=1000, axis=axis, dephase=False, zi=None)
        oresult = np.concatenate([arr for arr in pro_filt], axis=axis)

        # filter with scipy
        spresult = sps.sosfilt(filt.coeffs, arr, axis=axis, zi=None)

        assert np.allclose(oresult, spresult)

    
def test_sosfiltfilt_pros():
    """Test if openseize's sosfiltfilt matches scipy's restults for
    a elliptical IIR using sos fmt for a variety of producer sizes."""

    rng = np.random.default_rng(0)
    axis = -1
    fs = 500

    lengths = rng.integers(10000, 230000, size=10) 

    for length in lengths:

        # build array and producer
        arr = rng.random((2, 3, length))
        pro = producer(arr, chunksize=10000, axis=axis)

        # build an openseize filter and call it
        filt = build_filter(iir.Butter, fpass=100, fstop=200, fs=fs)
        pro_filt = filt(pro, chunksize=1000, axis=axis, dephase=True)
        oresult = np.concatenate([arr for arr in pro_filt], axis=axis)

        # filter with scipy
        spresult = sps.sosfiltfilt(filt.coeffs, arr, axis=axis, 
                                   padtype=None)

        assert np.allclose(oresult, spresult)


def test_sosfiltfilt_chunksizes():
    """Test if openseizes sosfiltfilt matches scipy's for a Chebyshev Type
    1 IIR using sos fmt for a variety of chunksizes."""

    rng = np.random.default_rng(9)
    axis=0
    fs = 2500

    arr = rng.random((101400, 2, 4))

    csizes= rng.integers(1000, 12300, size=9)

    # build an openseize filter
    filt = build_filter(iir.Cheby1, fpass=[200,600], fstop=[150, 650], 
                            fs=fs)
    for csize in csizes:

        pro = producer(arr, chunksize=csize, axis=axis)

        pro_filt = filt(pro, chunksize=csize, axis=axis, dephase=True)
        oresult = np.concatenate([arr for arr in pro_filt], axis=axis)

        # filter with scipy
        spresult = sps.sosfiltfilt(filt.coeffs, arr, axis=axis, 
                                   padtype=None)

        assert np.allclose(oresult, spresult, atol=1e-4)


def test_ba_coeffs0():
    """Determines if openseizes 'ba' coeffecients for a filter matches scipys
    iirdesign for a variety of iir filters."""

    # build an openseize filter & fetch coeffs
    fpass = np.array([900, 1500])
    fstop = np.array([400, 2000])
    gpass = 2
    gstop = 40
    fs = 5000

    types = iir.Butter, iir.Cheby1, iir.Cheby2, iir.Ellip

    for ftype in types:
       
        # build openseize filter and get its sos coeffs
        filt = build_filter(ftype, fpass, fstop, gpass=gpass, gstop=gstop,
                            fs=fs, fmt='ba')

        o_ba = filt.coeffs

        # build scipy filter and get its sos coeffs
        sp_ba = sps.iirdesign(2*fpass/fs, 2*fstop/fs, gpass=gpass, gstop=gstop,
                               ftype=ftype.__name__.lower(),
                               output='ba')

        assert np.allclose(o_ba, sp_ba)


def test_ba_coeffs1():
    """Test if openseize's ba coeffs match scipys for a variety of pass and
    stop bands."""

    # lowpass, bandpass highpass
    fpasses = [300, np.array([200, 450]), 1000]
    fstops = [400, np.array([100, 600]), 800]

    gpass, gstop = 1, 30
    fs = 5000

    for fpass, fstop in zip(fpasses, fstops):
        
        # build openseize filter and get its sos coeffs
        filt = build_filter(iir.Butter, fpass, fstop, gpass=gpass, 
                            gstop=gstop,fs=fs, fmt='ba')
        
        o_ba = filt.coeffs

        # build scipy filter and get its sos coeffs
        sp_ba = sps.iirdesign(2*fpass/fs, 2*fstop/fs, gpass=gpass, 
                               gstop=gstop, ftype='butter', output='ba')

        assert np.allclose(o_ba, sp_ba)


def test_lfilter_pros():
    """Test if openseizes lfilter for a Butterworth filter using 'ba' fmt
    coeffs matches scipy's result for a variety of producer sizes."""

    rng = np.random.default_rng(0)
    axis = 0
    fs = 500

    lengths = rng.integers(10000, 23000, size=10) 

    for length in lengths:
        
        # build array and producer
        arr = rng.random((length, 4, 6))
        pro = producer(arr, chunksize=1000, axis=axis)

        # build an openseize filter and call it
        filt = build_filter(iir.Butter, fpass=100, fstop=200, fs=fs,
                            fmt='ba')
        pro_filt = filt(pro, chunksize=1000, axis=axis, dephase=False)
        oresult = np.concatenate([arr for arr in pro_filt], axis=axis)

        # filter with scipy
        spresult = sps.lfilter(*filt.coeffs, arr, axis=axis, zi=None)

        assert np.allclose(oresult, spresult)


def test_lfilter_chunksizes():
    """Test that openseizes lfilter matches scipys for a variety of
    chunksizes using a Chebyshev type I filter and 'ba' fmt."""

    rng = np.random.default_rng(9)
    axis=-1
    fs = 2500

    arr = rng.random((2, 4, 232054))

    csizes= rng.integers(1000, 123000, size=10)

    for csize in csizes:

        pro = producer(arr, chunksize=csize, axis=axis)

        # build an openseize filter and call it
        filt = build_filter(iir.Cheby1, fpass=[200,600], fstop=[150, 650], 
                            fs=fs, fmt='ba')
        pro_filt = filt(pro, chunksize=csize, axis=axis, dephase=False)
        oresult = np.concatenate([arr for arr in pro_filt], axis=axis)

        # filter with scipy
        spresult = sps.lfilter(*filt.coeffs, arr, axis=axis)

        assert np.allclose(oresult, spresult)


def test_filtfilt_pros():
    """Test if openseize's filtfilt matches scipy's restults for
    a elliptical IIR using ba fmt for a variety of producer sizes."""

    rng = np.random.default_rng(0)
    axis = 1
    fs = 500

    lengths = rng.integers(10000, 230000, size=10) 

    for length in lengths:

        # build array and producer
        arr = rng.random((3, length, 4))
        pro = producer(arr, chunksize=10000, axis=axis)

        # build an openseize filter and call it
        filt = build_filter(iir.Ellip, fpass=100, fstop=200, fs=fs,
                            fmt='ba')
        pro_filt = filt(pro, chunksize=10000, axis=axis, dephase=True)
        oresult = np.concatenate([arr for arr in pro_filt], axis=axis)

        # filter with scipy
        spresult = sps.filtfilt(*filt.coeffs, arr, axis=axis, 
                                   padtype=None)

        assert np.allclose(oresult, spresult)


def test_filtfilt_chunksizes():
    """Test if openseize filtfilt matches scipy for a variety of
    chunksizes."""

    rng = np.random.default_rng(9)
    axis=-1
    fs = 3000

    arr = rng.random((2, 4, 123200))

    csizes= rng.integers(1000, 123000, size=10)

    for csize in csizes:

        pro = producer(arr, chunksize=csize, axis=axis)

        # build an openseize filter and call it
        filt = build_filter(iir.Butter, fpass=[300,900], fstop=[150, 1050], 
                            fs=fs, fmt='ba')
        pro_filt = filt(pro, chunksize=csize, axis=axis, dephase=True)
        oresult = np.concatenate([arr for arr in pro_filt], axis=axis)

        # filter with scipy
        spresult = sps.filtfilt(*filt.coeffs, arr, axis=axis, padtype=None)

        assert np.allclose(oresult, spresult)


def test_Notch_coeffs():
    """Test if the 'ba' coeffs of the openseize Notch match the 'ba' coeffs
    of scipy's iirNotch for a range of stops, widths."""

    fs = 500
    params = [(60, 4), (21.2, 2), (80.5, 10), (55, 3), (99, 1), (120.5, 33)]

    for tup in params:
        
        filt = build_filter(iir.Notch, *tup, fs=fs)
        o_b, o_a = filt.coeffs

        sp_b, sp_a = sps.iirnotch(tup[0], tup[0]/tup[1], fs=fs)
        assert(np.allclose(o_b, sp_b))
        assert(np.allclose(o_a, sp_a))


def test_Notch_pros():
    """Test if openseize Notch filter matches scipy notch for
    a forward-backward filter in 'ba' fmt."""

    
    rng = np.random.default_rng(0)
    axis = -1
    fs = 500

    lengths = rng.integers(10000, 230000, size=10) 

    for length in lengths:

        # build array and producer
        arr = rng.random((6, 1, length))
        pro = producer(arr, chunksize=1000, axis=axis)

        filt = build_filter(iir.Notch, fstop=60, width=4, fs=fs)
        pro_filt = filt(pro, chunksize=1000, axis=axis, dephase=True)
        oresult = np.concatenate([arr for arr in pro_filt], axis=axis)

        # filter with scipy
        spresult = sps.filtfilt(*filt.coeffs, arr, axis=axis, padtype=None)

        assert np.allclose(oresult, spresult)


def test_Notch_pros_nophase():
    """Test if openseize Notch filter matches scipy notch for
    a forward-backward filter in 'ba' fmt. for a variety of producer sizes
    without dephasing."""

    
    rng = np.random.default_rng(0)
    axis = -1
    fs = 500

    lengths = rng.integers(10000, 230000, size=10) 

    for length in lengths:

        # build array and producer
        arr = rng.random((6, 1, length))
        pro = producer(arr, chunksize=1000, axis=axis)

        filt = build_filter(iir.Notch, fstop=60, width=4, fs=fs)
        pro_filt = filt(pro, chunksize=1000, axis=axis, dephase=False)
        oresult = np.concatenate([arr for arr in pro_filt], axis=axis)

        # filter with scipy
        spresult = sps.lfilter(*filt.coeffs, arr, axis=axis)

        assert np.allclose(oresult, spresult)





if __name__ == '__main__':

    import matplotlib.pyplot as plt
    oresult, spresult, filt = test_sosfiltfilt_pros()

    plt.plot(oresult[0,0,:])
    plt.plot(spresult[0,0,:])
    plt.show()
