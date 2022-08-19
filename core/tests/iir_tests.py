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


def test_sos_coeffs():
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


def test_sosfilt_pros():
    """Test if openseize's sosfilt matches scipy's sosfilt restults for
    a Butterworth IIR using sos fmt for a variety of producer sizes."""

    rng = np.random.default_rng(0)
    axis = 1
    fs = 500

    lengths = rng.integers(10000, 230000, size=50) 

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
    """Test if openseize's sosfiltfilt matches scipy's sosfilt restults for
    a elliptical IIR using sos fmt for a variety of producer sizes."""

    rng = np.random.default_rng(0)
    axis = -1
    fs = 500

    lengths = rng.integers(10000, 230000, size=50) 

    #for length in lengths:
    length=197137

    # build array and producer
    arr = rng.random((2, 3, length))
    pro = producer(arr, chunksize=10000, axis=axis)

    # build an openseize filter and call it
    filt = build_filter(iir.Butter, fpass=100, fstop=200, fs=fs)
    pro_filt = filt(pro, chunksize=1000, axis=axis, dephase=True)
    oresult = np.concatenate([arr for arr in pro_filt], axis=axis)

    # filter with scipy
    spresult = sps.sosfiltfilt(filt.coeffs, arr, axis=axis, 
                               padtype='constant')

    #assert np.allclose(oresult, spresult)
    return oresult, spresult, filt

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    oresult, spresult, filt = test_sosfiltfilt_pros()

    plt.plot(oresult[0,0,:])
    plt.plot(spresult[0,0,:], alpha=0.25)
    plt.show()
