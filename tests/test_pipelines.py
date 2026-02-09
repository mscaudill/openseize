"""A module for testing openseize pipeline callable.

Typical usage example:
    !pytest test_pipelines.py::<TEST NAME>
"""

import pickle
from itertools import permutations

import pytest
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
