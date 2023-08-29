"""A module for testing protools that manipulate the size, shape and values
produced by a producer.

Typical usage example:
    !pytest test_operatives.py::<TEST NAME>
"""

import pytest
import numpy as np
from itertools import permutations
from pytest_lazyfixture import lazy_fixture

from openseize import producer
from openseize.core import protools


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
    yield np.transpose(rng.random((9013, 6, 3)), axes=axes)

@pytest.fixture(scope='module', params=permutations(range(4)))
def random4D(rng, request):
    """Returns random 4D arrays with samples along each axis."""

    axes = request.param
    yield np.transpose(rng.random((10012, 2, 3, 3)), axes=axes)


# use lazy fixtures to pass parameterized fixtures into test
@pytest.mark.parametrize('arr', 
    [
        lazy_fixture('random1D'), 
        lazy_fixture('random2D'), 
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_production_pad(arr):
    """Test if a producer created by padding a producer along its production
    axis produces the correct ndarrays."""

    # build a producer from arr
    pro_axis = np.argmax(arr.shape)
    pro = producer(arr, chunksize=1000, axis=pro_axis)

    # pad the producer
    amt = (10, 752)
    padded_pro = protools.pad(pro, amt=amt, axis=pro_axis)
    
    # build a producer from the padded array to compare against
    pads = [(0,0) for _ in range(arr.ndim)]
    pads[pro_axis] = amt
    padded_arr = np.pad(arr, pads)
    ground_truth_pro = producer(padded_arr, chunksize=1000, axis=pro.axis)
    
    # test equivalency of padded producer with producer of padded array
    for arr, actual in zip(padded_pro, ground_truth_pro):
        assert np.allclose(arr, actual)
  

@pytest.mark.parametrize('arr', 
    [
        lazy_fixture('random1D'), 
        lazy_fixture('random2D'), 
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_nonproduction_pad(arr):
    """Validate that padding a producer along any non-production axis produces
    the correct ndarrays."""

    pro_axis = np.argmax(arr.shape)
    pro = producer(arr, chunksize=1000, axis=pro_axis)

    # allow padding axes to vary over every non-production axis
    padding_axes = [ax for ax in range(arr.ndim) if ax != pro_axis]
    amt = (190, 13)

    # create padded_producers
    padded_pros = [protools.pad(pro, amt=amt, axis=ax) for ax in padding_axes]

    # create ground truth padded array producers
    ground_truth_pros = []
    for ax in padding_axes:

        # build pad for this non-production axis
        pads = [(0,0) for _ in range(arr.ndim)]
        pads[ax] = amt

        # build producer from padded array
        x = np.pad(arr, pads)
        ground_truth_pros.append(producer(x, chunksize=1000, axis=pro.axis))

    for padded_pro, gt_pro in zip(padded_pros, ground_truth_pros):
        for arr, actual in zip(padded_pro, gt_pro):
            assert np.allclose(arr, actual)
    
    
@pytest.mark.parametrize('arr', 
    [
        lazy_fixture('random1D'), 
        lazy_fixture('random2D'), 
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_expand_dims(arr):
    """Test if expand_dims inserts an axis at the correct position in an
    expanded producer."""

    axis = np.argmax(arr.shape)
    pro = producer(arr, chunksize=1000, axis=axis)
    
    for insertion in range(arr.ndim):
        expanded = protools.expand_dims(pro, axis=insertion)
        
        for x, y in zip(pro, expanded):
            
            shape = list(x.shape)
            shape.insert(insertion, 1)
            assert shape == list(y.shape)


@pytest.mark.parametrize('arr', 
    [
        lazy_fixture('random2D'), 
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_multiply_along_axis(arr):
    """Validates that multiplying a producer along all possible axes produces
    the correct produced arrays."""

    pro_axis = np.argmax(arr.shape)
    pro = producer(arr, chunksize=1000, axis=pro_axis)
    
    multiplier = 4.3 * np.ones(pro.shape[0])
    # call multiply along 0th axis for each transposed arr
    result = protools.multiply_along_axis(pro, multiplier, axis=0).to_array()

    #broadcast multiplier for multiplication along 0th axis
    shape = np.ones(arr.ndim, dtype=int)
    shape[0] = len(multiplier)
    broadcasted = multiplier.reshape(shape)
    ground_truth = arr * broadcasted

    assert np.allclose(result, ground_truth)


@pytest.mark.parametrize('arr', 
    [
        lazy_fixture('random1D'),
        lazy_fixture('random2D'), 
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_slice_along_axis(arr):
    """Validates that slicing along all possible axes produces the correct
    produced arrays."""

    pro_axis = np.argmax(arr.shape)
    pro = producer(arr, chunksize=1000, axis=pro_axis)

    # if production axis is along 0th take a bigger slice
    if pro_axis == 0:
        start, stop, step = 1203, 4309, 3
    # if not production axis take smaller slice
    else:
        start, stop, step = 0, 2, None

    # slice and convert to array
    result = protools.slice_along_axis(pro, start, stop, step, axis=0).to_array()

    assert np.allclose(arr[start:stop:step], result)
