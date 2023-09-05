"""A module for testing arraytools that manipulate the size and values of
ndarrays."""

import numpy as np
import pytest
from itertools import permutations
from pytest_lazyfixture import lazy_fixture

from openseize.core import arraytools


@pytest.fixture(scope='module')
def rng():
    """Returns a numpy default_rng instance for generating random but
    reproducible ndarrays."""

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
    yield np.transpose(rng.random((9013, 6, 4)), axes=axes)

@pytest.fixture(scope='module', params=permutations(range(4)))
def random4D(rng, request):
    """Returns random 4D arrays with samples along each axis."""

    axes = request.param
    yield np.transpose(rng.random((10012, 7, 4, 4)), axes=axes)

# use lazy fixtures to pass parameterized fixtures into test
@pytest.mark.parametrize('arr',
    [
        lazy_fixture('random1D'),
        lazy_fixture('random2D'),
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_slice_along_axis(arr):
    """Verifies that slice_along_axis produces the correct arrays for inputs of
    1-4 dimensions."""

    # for testing slice along last axis
    axis = -1
    min_length = min(arr.shape)
    start, stop, step = 2, None, 7
    x = arraytools.slice_along_axis(arr, start, stop, step, axis=axis)
    y = arr[..., start:stop:step]
    assert np.allclose(x, y)

@pytest.mark.parametrize('arr',
    [
        lazy_fixture('random1D'),
        lazy_fixture('random2D'),
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_pad_axis(arr):
    """Verifies the correct number of zero pads are applied along the longest
    axis of an arr."""

    pad = [43, 70]
    axis = np.argmax(arr.shape)
    x = arraytools.pad_along_axis(arr, pad=pad, axis=axis)

    # slice off the zeros and test equality with input
    start, stop = pad[0], x.shape[axis] - pad[1]
    x = arraytools.slice_along_axis(x, start, stop, axis=axis)
    assert np.allclose(x, arr)

@pytest.mark.parametrize('arr',
    [
        lazy_fixture('random1D'),
        lazy_fixture('random2D'),
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_split_along_axis(arr):
    """Verifies that splitting along an axis returns two correct valued
    arrays."""

    # for testing slice along the longest
    axis = np.argmax(arr.shape)
    index = 300
    a, b = arraytools.split_along_axis(arr, index, axis=axis)
    c, d = np.split(arr, [index], axis=axis)
    assert np.allclose(a, c)
    assert np.allclose(b, d)

@pytest.mark.parametrize('arr',
    [
        lazy_fixture('random1D'),
        lazy_fixture('random2D'),
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_expand(arr):
    """Verifies that expansion by insertion of samples along an axis."""

    axis = np.argmax(arr.shape)
    l, value = 10, -10
    x = arraytools.expand_along_axis(arr, l=l, value=value, axis=axis)
    y = arraytools.slice_along_axis(x, None, None, l, axis=axis)
    assert np.allclose(y, arr)

@pytest.mark.parametrize('arr',
    [
        lazy_fixture('random1D'),
        lazy_fixture('random2D'),
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_multiply_along_axis(arr):
    """Verifies multiplication of an ndarray by a 1-D array along an axis."""

    # multiply along the longest
    axis = np.argmax(arr.shape)
    multiplier = 2 * np.ones(arr.shape[axis])
    x = arraytools.multiply_along_axis(arr, multiplier, axis=axis)
    assert np.allclose(x - arr, arr)

def test_filter1D():
    """Test that filter1D returns the expected boolean array."""

    size=100
    indices = np.random.choice(np.arange(size), size=10, replace=False)
    x = arraytools.filter1D(size, indices)
    assert np.allclose(np.where(x)[0], np.sort(indices))
