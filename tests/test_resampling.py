"""A module for testing the resampling arrays and producer of arrays.

Typical usage example:
    !pytest resampling_test.py::<TEST_NAME>
"""

import pytest
import itertools
import numpy as np
import scipy.signal as sps

from openseize import producer
from openseize.resampling.resampling import resample
from openseize.filtering.fir import Kaiser

def random_arr(shape=(6, 39968), seed=33):
    """Returns a random array of shape."""

    rng = np.random.default_rng(seed)
    return rng.random(shape)


def resample_filter(L, M, fs, gpass=0.1, gstop=40):
    """Returns a Kaiser Interpolating/Antialiasing filter matching
    Openseize's default filter."""

    g = np.gcd(L, M)
    L //= g
    M //= g
    fcut = fs / (2 * max(L,M))
    fstop = fcut + fcut / 10
    fpass = fcut - fcut / 10
    gpass, gstop = .1, 40
    h = Kaiser(fpass, fstop, fs, gpass, gstop).coeffs

    return h


def test_LM():
    """Test if iterative polyphase decomposition in openseize resample
    matches scipy resample results for combinations of the up and downsample
    factors."""

    #build random array & set chunksize and fs
    rarr = random_arr()
    chunksize = 120230
    fs = 5000

    # make permutations of all L and M in [1, 13]
    for L, M in itertools.permutations(np.arange(1, 6), r=2):
        
        # iteratively compute using openseize resampling
        x = resample(rarr, L=L, M=M, fs=fs, chunksize=chunksize, axis=-1)

        # build the same filter for scipy that openseize uses
        h = resample_filter(L, M, fs)

        # call scipy resample and verify equivalence
        y = sps.resample_poly(rarr, up=L, down=M, axis=-1, window=h)

        assert np.allclose(x,  y)


def test_chunksizes():
    """Test if iterative polyphase decomposition in openseize resample
    matches scipy resample for a variety of chunksizes."""

    # construct random array & set an fs, L and M.
    rarr = random_arr()
    fs = 3200
    L = 3
    M = 11

    # construct a range of chunksizes
    rng = np.random.default_rng(33)
    chunksizes = rng.choice(np.arange(1000, 500000), size=20)
    
    for csize in chunksizes:

        # iteratively compute using openseize resampling
        x = resample(rarr, L=L, M=M, fs=fs, chunksize=csize, axis=-1)

        # build the same filter for scipy that openseize uses
        h = resample_filter(L, M, fs)

        # call scipy resample and verify equivalence
        y = sps.resample_poly(rarr, up=L, down=M, axis=-1, window=h)
        
        assert np.allclose(x,  y)


def test_sizes():
    """Test if iterative polyphase decomposition in openseize resample
    matches scipy resample for a variety of array sizes."""

    # create random array of samples between 100K and 1M samples
    rng = np.random.default_rng(311)
    samples = rng.integers(low=int(1e5), high=int(5e5), size=10) 

    # create random array of channel counts
    chs = rng.integers(low=1, high=17, size=20)
    
    # set all other parameters
    fs = 2000
    L, M = 2, 7
    chunksize = 19089

    # build the same filter for scipy that openseize uses
    h = resample_filter(L, M, fs)

    for nchs, nsamples in zip(chs, samples):

        # create an random array of nsamples size along last axis
        rarr = rng.random((nchs, nsamples))
        
        # iteratively compute using openseize resampling
        x = resample(rarr, L=L, M=M, fs=fs, chunksize=chunksize, axis=-1)

        # compute with scipy
        y = sps.resample_poly(rarr, up=L, down=M, axis=-1, window=h)

        assert np.allclose(x,  y)


def test_LM1():
    """Test if L==M, the array supplied to Openseize is the array
    returned."""

    # construct random array & set an fs, L and M.
    rarr = random_arr()
    chunksize=10002
    fs = 3200
    L = 11
    M = 11

    # iteratively compute using openseize resampling
    x = resample(rarr, L=L, M=M, fs=fs, chunksize=chunksize, axis=-1)
    
    assert np.allclose(x, rarr)


