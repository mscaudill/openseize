"""A module for testing the resampling arrays and producer of arrays.

Typical usage example:
    !pytest resampling_test::<TEST_NAME>
"""

import pytest
import itertools
import numpy as np
import scipy.signal as sps

from openseize import producer
from openseize.resampling.resampling import resample
from openseize.filtering.fir import Kaiser

def test_resampling_factors():
    """Test if iterative polyphase decomposition in openseize resample
    matches scipy resample results for 66 unique combinations of up and
    downsample factors L and M."""

    rng = np.random.default_rng(33)
    arr = rng.random((4, 3996797))
    chunksize = 120237
    fs = 5000

    # 12 choose 2 combinations of L and M = 66 
    for L, M in itertools.combinations(np.arange(1, 13), r=2):
        
        # iteratively compute using openseize resampling
        x = resample(arr, L=L, M=M, fs=fs, chunksize=chunksize, axis=-1)

        # compute with scipy
        fstop = fs // max(L, M)
        fpass = fstop - fstop//10
        gpass, gstop = 1, 40
        h = Kaiser(fpass, fstop, fs, gpass, gstop).coeffs
        y = sps.resample_poly(arr, up=L, down=M, axis=-1, window=h)

        assert np.allclose(x,  y)

def test_chunksizes():
    """Test if iterative polyphase decomposition in openseize resample
    matches scipy resample for a variety of chunksizes."""

    rng = np.random.default_rng(33)
    chunksizes = rng.choice(np.arange(1000, 1500000), size=20)
    
    arr = rng.random((4, 3996790))
    fs = 4096
    L = 1
    M = 11

    for csize in chunksizes:

        # iteratively compute using openseize resampling
        x = resample(arr, L=L, M=M, fs=fs, chunksize=csize, axis=-1)

        # compute with scipy
        fstop = fs // max(L, M)
        fpass = fstop - fstop//10
        gpass, gstop = 1, 40
        h = Kaiser(fpass, fstop, fs, gpass, gstop).coeffs
        y = sps.resample_poly(arr, up=L, down=M, axis=-1, window=h)

        assert np.allclose(x,  y)

def test_sizes():
    """Test if iterative polyphase decomposition in openseize resample
    matches scipy resample for a variety of array sizes."""

    rng = np.random.default_rng(311)
    #  vary num samples in array along last axis
    samples = rng.integers(low=int(10e6), high=int(20e6), size=20) 
    
    fs = 2000
    L = 2
    M = 7
    chunksize = 19089

    for nsamples in samples:

        arr = rng.random((4, nsamples))
        
        # iteratively compute using openseize resampling
        x = resample(arr, L=L, M=M, fs=fs, chunksize=chunksize, axis=-1)

        # compute with scipy
        fstop = fs // max(L, M)
        fpass = fstop - fstop//10
        gpass, gstop = 1, 40
        h = Kaiser(fpass, fstop, fs, gpass, gstop).coeffs
        y = sps.resample_poly(arr, up=L, down=M, axis=-1, window=h)

        assert np.allclose(x,  y)




