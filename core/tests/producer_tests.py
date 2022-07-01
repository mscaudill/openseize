"""A module for testing Producer instances built from arrays, sequences,
generators, file Readers with and without boolean masks.

Typical usage example:
    !pytest producer_tests.py::<TEST_NAME>
"""

import pytest
import numpy as np
from itertools import zip_longest
from pathlib import Path

from openseize.core.producer import producer, as_producer, Producer
from openseize.io.edf import Reader
from openseize import demos

def test_fromarray():
    """Verify that a producer built from an array produces the correct
    subarrays on each iteration."""

    # build random data array and producer
    arr = np.random.random((119, 510023))
    chunksize = 2031
    pro = producer(arr, chunksize=chunksize, axis=-1)
    # compare each produced array against sliced out subarray
    starts = range(0, arr.shape[-1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[-1])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[-1] = slice(start, stop)
        assert np.allclose(arr[tuple(slicer)], pro_arr)

def test_frommaskedarray():
    """Test if a producer from arrays with a mask produces correct subarrays
    on each iteration."""

    # create a reproducible random array and mask to apply along axis=-1
    rng = np.random.default_rng(seed=0)
    arr = rng.random((17, 12, 52010))
    mask = rng.choice([True, False], size=arr.shape[-1], p=[.2, .8])
    # mask the array
    masked = arr[:,:, mask]
    # build a masked producer
    chunksize = 10000
    pro = producer(arr, chunksize=chunksize, axis=-1, mask=mask)

    # test equality
    starts = range(0, masked.shape[-1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=masked.shape[-1])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * masked.ndim #slice obj to slice arr
        slicer[-1] = slice(start, stop)
        assert np.allclose(masked[tuple(slicer)], pro_arr)

def test_fromsequence():
    """Verify that a producer built from a sequence of arrays produces the
    correct subarrays on each iteration."""

    # build a sequence of random arrays that vary in samples along 1st axis
    arrs = [np.random.random((3, x, 4)) 
            for x in np.random.randint(1000, 100000, size=16)]
    # build full array for easy slicing
    arr  = np.concatenate(arrs, axis=1)
    # build producer from the sequence of ndarrays
    chunksize = 1280
    pro = producer(arrs, chunksize=chunksize, axis=1)
    # test equality
    starts = range(0, arr.shape[1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[1])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[1] = slice(start, stop)
        assert np.allclose(arr[tuple(slicer)], pro_arr)

def test_fromgenerator0():
    """Verify that a producer built from a generator function yielding
    subarrays of varying lengths along chunk axis produces the correct 
    subarrays on each iteration."""

    # build random test arrays and concatenate for testing
    lens = np.random.randint(2000, high=80034, size=50)
    arrs = [np.random.random((l, 2, 17)) for l in lens]
    arr = np.concatenate(arrs, axis=0)

    def g(arrs):
        """Generating function of random arrays."""

        for arr in arrs:
            yield arr

    chunksize=20000
    # create a producer from the generator func g passing in arrs as kwargs
    pro = producer(g, chunksize=chunksize, axis=0, shape=arr.shape,
                   arrs=arrs)
    # test equality
    starts = range(0, arr.shape[0], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[0])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[0] = slice(start, stop)
        assert np.allclose(arr[tuple(slicer)], pro_arr)

def test_fromgenerator1():
    """Test if a producer with a chunksize that is a multiple of the array
    size (along chunk axis) yielded by a generator yields the correct 
    subarrays on each iteration.
    """

    # set size of data created by generator
    chs = 7
    samples = 501000
    gensize = 10000 #number of samples in each yielded array

    # set the chunksize multiple of the gensize that will be produced
    multiple = 3
    chunksize = multiple * gensize

    def g(chs=chs, samples=samples, segsize=gensize):
        """A generator of constant sized arrays of segsize along axis=-1.

        chs: int
            The size of each yielded array on 0th axis.
        samples: int 
            The total size of all yielded arrays along 1st axis.
        segsize:
            The number of samples along 1st axis yielded  per iteration of
            this generating function.
        """

        rng = np.random.default_rng(seed=1)
        starts = range(0, samples, segsize)
        segments = zip_longest(starts, starts[1:], fillvalue=samples)
        for start, stop in segments:
            arr = rng.random((chs, stop-start))
            yield arr

    # build a producer
    pro = producer(g, chunksize=chunksize, axis=-1, 
                   shape=(chs, samples))
    # combine all the arrays from the generator for easy testing
    arr = np.concatenate([x for x in g()], axis=-1)
    # test equality
    starts = range(0, arr.shape[-1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[-1])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[-1] = slice(start, stop)
        assert np.allclose(arr[tuple(slicer)], pro_arr)

def test_frommaskedgenerator0():
    """Verify that a producer from a generator of arrays with a mask
    produces the correct subarrays on each iteration."""

    # build random test arrays and concatenate for testing
    lens = np.random.randint(21000, high=80034, size=151)
    arrs = [np.random.random((l, 4, 9)) for l in lens]
    arr = np.concatenate(arrs, axis=0)

    # create a generator of the random array sequence
    def g(arrs):
        """Generating function of random arrays."""

        for arr in arrs:
            yield arr

    # create a mask and masked producer
    rng = np.random.default_rng(seed=0)
    mask = rng.choice([True, False], size=arr.shape[0], p=[.3, .7])
    # mask the array
    masked = arr[mask, :, :]
    chunksize = 10011
    pro = producer(arr, chunksize=chunksize, axis=0, mask=mask)
    
    # test equality
    starts = range(0, masked.shape[0], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=masked.shape[0])
    for idx, ((start, stop), pro_arr) in enumerate(zip(segments, pro)):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[0] = slice(start, stop)
        assert np.allclose(masked[tuple(slicer)], pro_arr)

def test_frommaskedgenerator1():
    """Verify that a producer from a generator of arrays with a mask
    produces the correct subarrays on each iteration when the chunksize is
    a multiple of the generator size along chunking axis."""


    # set size of data created by generator
    chs = 6
    samples = 5201000
    gensize = 10002 #number of samples in each yielded array

    # set the chunksize multiple of the gensize that will be produced
    multiple = 3
    chunksize = multiple * gensize

    def g(chs=chs, samples=samples, segsize=gensize):
        """A generator of constant sized arrays of segsize along axis=-1.

        chs: int
            The size of each yielded array on 0th axis.
        samples: int 
            The total size of all yielded arrays along 1st axis.
        segsize:
            The number of samples along 1st axis yielded  per iteration of
            this generating function.
        """

        rng = np.random.default_rng(seed=1)
        starts = range(0, samples, segsize)
        segments = zip_longest(starts, starts[1:], fillvalue=samples)
        for start, stop in segments:
            arr = rng.random((chs, stop-start))
            yield arr

    # combine all the arrays from the generator for easy testing
    arr = np.concatenate([x for x in g()], axis=-1)

    #build a mask and apply it to arr
    rng = np.random.default_rng(seed=0)
    mask = mask = rng.choice([True, False], size=arr.shape[-1], p=[.3, .7])
    masked = arr[:, mask]

    # build a producer
    pro = producer(g, chunksize=chunksize, axis=-1, 
                   shape=(chs, samples), mask=mask)
    # test equality
    starts = range(0, masked.shape[-1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[-1])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[-1] = slice(start, stop)
        assert np.allclose(masked[tuple(slicer)], pro_arr)

def test_fromreader():
    """Verifies that a producer from an EDF file reader produces the correct
    subarrays in each iteration."""

    fp = demos.paths['M5872_Left_group_A.edf']
    reader = Reader(fp)
    # read all the data of this small file
    arr = reader.read(0)

    #build a producer and test equality
    chunksize=10023
    pro = producer(reader, chunksize=chunksize, axis=-1)
    starts = range(0, arr.shape[-1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[-1])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[-1] = slice(start, stop)
        assert np.allclose(arr[tuple(slicer)], pro_arr)

def test_frommaskedreader():
    """Verify that a producer from an EDF file reader with a mask produces
    the correct subarrays in each iteration."""

    fp = demos.paths['M5872_Left_group_A.edf']
    reader = Reader(fp)
    # read all the data of this small file
    arr = reader.read(0)

    #build a mask and apply it to the array
    rng = np.random.default_rng(seed=0)
    mask = mask = rng.choice([True, False], size=arr.shape[-1], p=[.8, .2])
    masked = arr[:, mask]

    #build a producer and test equality
    chunksize=20000
    pro = producer(reader, chunksize=chunksize, axis=-1, mask=mask)
    starts = range(0, masked.shape[-1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=masked.shape[-1])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[-1] = slice(start, stop)
        assert np.allclose(masked[tuple(slicer)], pro_arr)

def test_asproducer0():
    """Verify that the as_producer decorator correctly decorates
    a generating function converting it into a producer type."""

    # build random test arrays and concatenate for testing
    lens = np.random.randint(21100, high=80092, size=101)
    arrs = [np.random.random((l, 4, 9)) for l in lens]

    pro = producer(arrs, chunksize=10000, axis=0)

    @as_producer
    def my_gen(pro):
        """A generating function yielding ndarrays."""

        for arr in pro:
            yield arr**2

    assert isinstance(my_gen(pro), Producer)

def test_asproducer1():
    """Verify's that arrays from a generating function converted to
    a producer yield the correct subarrays on each iteration."""

    # build random test arrays and concatenate for testing
    lens = np.random.randint(21100, high=80092, size=68)
    arrs = [np.random.random((l, 4, 9)) for l in lens]
    arr = np.concatenate(arrs, axis=0)

    pro = producer(arrs, chunksize=10000, axis=0)

    @as_producer
    def avg_gen(pro):
        """An averager that averages every 400 samples of the produced
        values."""
        
        # temporarily change the chunksize and average
        pro.chunksize = 400
        for arr in pro:
            yield np.mean(arr, axis=0, keepdims=True)

    # Build the ground truth array where we have averaged every 400-samples
    mstarts = range(0, arr.shape[0], 400)
    msegments = zip_longest(mstarts, mstarts[1:], fillvalue=arr.shape[0])
    means = []
    for start, stop in msegments:
        means.append(np.mean(arr[start:stop,:,:], axis=0, keepdims=True))
    meaned = np.concatenate(means, axis=0)

    # test equality of meaned with producer for each iteration
    # The as_producer decorator will set the chunksize back to the
    # producers original chunksize (i.e. 10000). This allows gen. functions
    # to change the chunksize for algorithmic efficiency but supply chunks
    # back to callers at the original requested size.
    starts = range(0, meaned.shape[0], 10000)
    segments = zip_longest(starts, starts[1:], fillvalue=meaned.shape[0])
    for (start, stop), pro_arr in zip(segments, my_gen(pro)):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[0] = slice(start, stop)
        assert np.allclose(meaned[tuple(slicer)], pro_arr)







