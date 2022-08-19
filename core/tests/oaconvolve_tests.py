"""A module for testing filtering tools and their subfunctions.

Typical usage example:
    !pytest oaconvolve_tests.py::<TEST_NAME>
"""

import pytest
import numpy as np
import scipy.signal as sps

from openseize import producer
from openseize.core import numerical as nm

def test_oaconvolve_pros():
    """Compares openseize's overlap-add convolve against scipy for
    arrays/producers that vary in size along the last axis."""

    # build 50 random lengths for arrays
    rng = np.random.default_rng(0)
    lengths = rng.integers(low=10000, high=700000, size=50)
    
    # convolve with a hann window of 203 samples
    window = sps.get_window('hann', 203)
    for l in lengths:

        # build arr and producer
        arr = np.random.random((3, 4, l)) 
        pro = producer(arr, chunksize=len(window), axis=-1)

        # compute openseize convolve
        osz_pro = nm.oaconvolve(pro, window, axis=-1, mode='same')
        osz_result = np.concatenate([x for x in osz_pro], axis=-1)

        # compute scipy convolve, window dims must match arr dims
        win = np.expand_dims(window, (0,1))
        sp_result = sps.oaconvolve(arr, win, axes=-1, mode='same')

        assert np.allclose(osz_result, sp_result)


def test_oaconvolve_mode():
    """Compares openseize's overlap-add convolve against scipy for each of
    the three possible convolve modes."""

    # build an 4-D array with samples along 1st axis and producer
    rng = np.random.default_rng(10)
    axis = 1
    arr = rng.random((4, 106453, 2, 3))
    pro = producer(arr, chunksize=10000, axis=axis)
    window = sps.get_window('blackman', 76)

    for mode in ('full','same','valid'):

        # compute openseize convolve
        osz_pro = nm.oaconvolve(pro, window, axis=axis, mode='same')
        osz_result = np.concatenate([x for x in osz_pro], axis=axis)

        # compute scipy convolve, window dims must match arr dims
        win = np.expand_dims(window, (0, 2, 3))
        sp_result = sps.oaconvolve(arr, win, axes=axis, mode='same')

        assert np.allclose(osz_result, sp_result)


def test_oaconvolve_windows():
    """Compares openseize's overlap-add convolve against scipy for a variety
    of scipy windows of various lengths."""

    rng = np.random.default_rng(10)

    lengths = rng.integers(low=50, high=300, size=25)
    windows = 'cosine bartlett boxcar blackman hann'.split()

    #build an 2-D array with samples along 0th axis
    axis=0
    mode = 'same'
    arr = rng.random((176221, 6))
    pro = producer(arr, chunksize=10000, axis=axis)

    for l in lengths:
        for win in windows:

            print(win)

            window = sps.get_window(win, l)

            # compute openseize convolve
            osz_pro = nm.oaconvolve(pro, window, axis=axis, mode=mode)
            osz_result = np.concatenate([x for x in osz_pro], axis=axis)

            # compute scipy convolve, window dims must match arr dims
            win = np.expand_dims(window, 1)
            sp_result = sps.oaconvolve(arr, win, axes=axis, mode=mode)

            assert np.allclose(osz_result, sp_result)








   


if __name__ == "__main__":

    ores, spres = test_oaconvolve_mode()
