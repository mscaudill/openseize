"""A module for testing filtering tools and their subfunctions.

Typical usage example:
    !pytest filtering_tests.py::<TEST_NAME>
"""


import pytest
import numpy as np
import scipy.signal as sps

from openseize import producer
from openseize.core import numerical as nm

def test_oaconvolve0():
    """Compares openseize's overlap-add convolve against scipy for
    arrays/producers of various sizes."""

    rng = np.random.default_rng(0)
    lengths = rng.integers(low=10000, high=700000, size=50)
    
    window = sps.get_window('hann', 200)
    for l in lengths:

        arr = np.random.random((3, 4, l)) 
        pro = producer(arr, chunksize=len(window), axis=-1)

        osz_pro = nm.oaconvolve(pro, window, axis=-1, mode='same')
        osz_result = np.concatenate([x for x in osz_pro], axis=-1)

        win = np.expand_dims(window, (0,1))
        sp_result = sps.oaconvolve(arr, win, axes=-1, mode='same')

        assert np.allclose(osz_result, sp_result)
   


if __name__ == "__main__":

    test_oaconvolve0()
