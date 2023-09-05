"""A module for testing concurrency with Openseize objects including readers,
producers, and pipelines of DSP operations.

Typical usage example:
    # -rA flag shows extra summary see pytest -h
    !pytest -rA test_concurrency::<TEST_NAME>
"""

import pickle
import time
import multiprocessing as mp
from functools import partial

import numpy as np
import pytest
from scipy.signal import windows

from openseize import producer
from openseize.core import numerical, resources
from openseize.demos import paths
from openseize.file_io import edf
from openseize.filtering import fir, iir
from openseize.resampling import resampling
from openseize.spectra.estimators import psd
from openseize.tools import pipeline


@pytest.fixture(scope="module")
def rng():
    """Returns a numpy default_rng object for generating reproducible but
    random ndarrays."""

    seed = 0
    return np.random.default_rng(seed)

@pytest.fixture(scope="module")
def random2D(rng):
    """Returns a random 2D array."""

    return rng.random((5, 17834000))

def genfunc(random2D, x=10):
    """A generating function for testing pickleability of producers built from
    generating functions."""

    for arr in random2D:
        yield arr * x

@pytest.fixture(scope='module')
def testpro(random2D):
    """A producer built from a random2D for testing with."""

    return producer(random2D, chunksize=3e5, axis=-1)

def test_edfreader(demo_data):
    """Validates that edf.Reader instances are picklable, a requirement for
    multiprocessing."""

    reader = edf.Reader(demo_data)
    pro = producer(reader, chunksize=30e6, axis=-1)
    assert resources.pickleable(pro)

def test_ArrayProducer(testpro):
    """Validates that a producer built from a sequence or Numpy array is
    pickleable."""

    assert resources.pickleable(testpro)

def test_GenProducer(random2D):
    """Validates that a producer built from a generating function is
    pickleable."""

    s = random2D.shape
    pro = producer(genfunc, chunksize=3e5, axis=-1, shape=s, x=8)
    assert resources.pickleable(pro)

def test_MaskedProducer(random2D):
    """Test that a MaskedProducer built from an ndarray and mask is
    pickleable."""

    mask = np.random.choice([True, False], random2D.shape[-1])
    pro = producer(random2D, chunksize=3e5, axis=-1, mask=mask)
    assert resources.pickleable(pro)

def test_oaconvolve(testpro):
    """Validate that all FIR filters relying oaconvovle are pickleable."""

    win = np.random.random(1000)
    genfunc = partial(numerical.oaconvolve, testpro, window=win, axis=-1, mode='same')
    pro = producer(genfunc, chunksize=3e5, axis=-1, shape=testpro.shape)
    assert resources.pickleable(pro)

def test_sosfilt(testpro):
    """Validate that all IIRs relying on sosfilt are pickleable."""

    filt = iir.Butter(300, 500, fs=5000)
    genfunc = partial(numerical.sosfilt, testpro, filt.coeffs, axis=-1)
    pro = producer(genfunc, chunksize=3e5, axis=-1, shape=testpro.shape)
    assert resources.pickleable(pro)

def test_sosfiltfilt(testpro):
    """Validate that all IIRs relying on sosfiltfilt are pickleable."""

    filt = iir.Butter(300, 500, fs=5000)
    genfunc = partial(numerical.sosfiltfilt, testpro, filt.coeffs, axis=-1)
    pro = producer(genfunc, chunksize=3e5, axis=-1, shape=testpro.shape)
    assert resources.pickleable(pro)

def test_lfilter(testpro):
    """Validate that all IIRs relying on lfilter are pickleable."""

    filt = iir.Butter(300, 500, fs=5000, fmt='ba')
    genfunc = partial(numerical.lfilter, testpro, filt.coeffs, axis=-1)
    pro = producer(genfunc, chunksize=3e5, axis=-1, shape=testpro.shape)
    assert resources.pickleable(pro)

def test_filtfilt(testpro):
    """Validate that all IIRs relying on lfilter are pickleable."""

    filt = iir.Butter(300, 500, fs=5000, fmt='ba')
    genfunc = partial(numerical.filtfilt, testpro, filt.coeffs, axis=-1)
    pro = producer(genfunc, chunksize=3e5, axis=-1, shape=testpro.shape)
    assert resources.pickleable(pro)

def test_polyphase(testpro):
    """Validate that all resamplers relying on polyphase_resampling are
    pickleable."""

    filt =  fir.Kaiser(300, 500, fs=5000)
    genfunc = partial(numerical.polyphase_resample, testpro, L=10, M=3, fs=5000,
                      fir=filt, axis=-1)
    pro = producer(genfunc, chunksize=3e5, axis=-1, shape=testpro.shape)
    assert resources.pickleable(pro)

def test_welch(testpro):
    """Validate that producers of Welch PSD estimates are pickleable."""

    freqs, pro = numerical.welch(testpro, fs=5000, nfft=512,
                      window=windows.hann(100), overlap=0.5, axis=-1, detrend=True,
                      scaling='density')
    assert resources.pickleable(pro)

def test_stft(testpro):
    """Validate that producers of STFT arrays are pickleable."""

    freqs, time, pro = numerical.stft(testpro, fs=5000, nfft=512,
                    window=windows.hann(100), overlap=0.5, axis=-1, detrend=True,
                    scaling='density', boundary=True, padded=True)
    assert resources.pickleable(pro)

def test_pipeline1():
    """Builds a downsampling and notch filtering pipeline to test
    pickleability."""

    pipe = pipeline.Pipeline()
    pipe.append(resampling.downsample, M=10, fs=5000, chunksize=3e5, axis=-1)
    notch = iir.Notch(60, width=6, fs=500)
    pipe.append(notch, chunksize=3e5, axis=-1)
    assert resources.pickleable(pipe)

def test_pipeline2():
    """Builds a downsampling followed by welch PSD estimate pipeline."""

    pipe = pipeline.Pipeline()
    pipe.append(resampling.downsample, M=10, fs=5000, chunksize=3e5, axis=-1)
    pipe.append(psd, fs=500)
    assert resources.pickleable(pipe)
