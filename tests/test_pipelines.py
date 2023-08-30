"""A module for testing openseize pipeline callable.

Typical usage example:
    !pytest test_pipelines.py::<TEST NAME>
"""

import pickle
from itertools import permutations

import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np
import scipy.signal as sps

from openseize import producer
from openseize.filtering.iir import Notch
from openseize.filtering.fir import Kaiser
from openseize.resampling.resampling import downsample
from openseize.tools.pipeline import Pipeline


@pytest.fixture(scope="module")
def rng():
    """Returns a numpy default_rng object for generating reproducible but
    random ndarrays."""

    seed = 0
    return np.random.default_rng(seed)

@pytest.fixture(scope="module")
def random1D(rng):
    """Returns a random 1D array."""

    return rng.random((230020,))

@pytest.fixture(scope='module', params=permutations(range(2)))
def random2D(rng, request):
    """Returns random 2D arrays with sample axis along each axis."""

    axes = request.param
    yield np.transpose(rng.random((100012, 6)), axes=axes)

@pytest.fixture(scope='module', params=permutations(range(3)))
def random3D(rng, request):
    """Returns random 3D arrays with samples along each axis."""

    axes = request.param
    yield np.transpose(rng.random((100012, 6, 3)), axes=axes)

@pytest.fixture(scope='module', params=permutations(range(4)))
def random4D(rng, request):
    """Returns random 4D arrays with samples along each axis."""

    axes = request.param
    yield np.transpose(rng.random((100012, 2, 3, 3)), axes=axes)

def test_validate():
    """Confirms TypeError when caller is appended with too few bound args."""

    pipe = Pipeline()

    def caller(a, b, c=0):
        return a + b + c

    with pytest.raises(TypeError) as exc:
        
        pipe.append(caller, c=10)
        assert exc.type is TypeError
        
def test_contains_functions():
    """Validates Pipeline's contain method for functions."""

    pipe = Pipeline()

    def f(a, b):
        return a+b

    def g(x, y=0):
        return x**y

    pipe.append(f, a=1, b=2)
    pipe.append(g, y=10)

    assert f in pipe
    assert g in pipe

def test_contains_callables():
    """Validates Pipeline's contain method for callables."""

    pipe = Pipeline()

    notch = Notch(60, width=6, fs=5000)
    pipe.append(notch, chunksize=10000, axis=-1)

    assert notch in pipe

# use lazy fixtures to pass parameterized fixtures into test
@pytest.mark.parametrize('arr', 
    [
        lazy_fixture('random1D'), 
        lazy_fixture('random2D'), 
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_call_method(arr):
    """Test that composed openseize callable return the same result as Scipy.

    This test is superfulous because all of openseize's functions and callables
    are tested in their respective testing modules (e.g. test_iir.py) but for
    completeness we test again.
    """
    
    axis = np.argmax(arr.shape)
    pro = producer(arr, chunksize=1000, axis=axis)
    
    # add notch & downsample
    pipe = Pipeline()
    notch = Notch(60, width=8, fs=1000)
    pipe.append(notch, chunksize=1000, axis=axis, dephase=False)
    pipe.append(downsample, M=10, fs=1000, chunksize=1000, axis=axis)

    measured = np.concatenate([x for x in pipe(pro)], axis=axis)

    # compare with scipy
    b, a = sps.iirnotch(60, Q=60/8, fs=1000) 
    notched = sps.lfilter(b, a, arr, axis=axis)

    # build a kaiser like the one openseize uses
    cutoff = 1000 / (2*10) # fs / 2M
    fstop = cutoff + cutoff / 10
    fpass = cutoff - cutoff / 10
    gpass, gstop = 0.1, 40
    h = Kaiser(fpass, fstop, 1000, gpass, gstop).coeffs

    downed = sps.resample_poly(notched, up=1, down=10, axis=axis, window=h)

    assert np.allclose(measured, downed)

def test_pickleable():
    """Test that pipelines are picklable. 

    Note this test is only designed to ensure pipelines are pickleable NOT that
    the contained callables are pickleable. For those test see test_concurrency.
    """

    pipe = Pipeline()

    # add notch & downsample
    pipe = Pipeline()
    notch = Notch(60, width=8, fs=5000)
    pipe.append(notch, chunksize=10000, axis=-1)
    pipe.append(downsample, M=10, fs=5000, chunksize=10000, axis=-1)

    # test pickling of this pipeline
    sbytes = pickle.dumps(pipe)
    assert isinstance(sbytes, bytes)

