"""A module for testing Producer instances built from arrays, sequences,
generators, file Readers with and without boolean masks.

Typical usage example:
    !pytest producer_tests.py::<TEST_NAME>
"""

import pytest
import time
import numpy as np
from itertools import zip_longest
from pathlib import Path

from openseize import producer
from openseize.core.producer import Producer
from openseize.file_io.edf import Reader
from openseize.core.arraytools import slice_along_axis


def test_fromarray():
    """Verify that a producer built from an array produces the correct
    subarrays on each iteration."""

    # build random data array and producer
    arr = np.random.random((119, 51002))
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
    lens = np.random.randint(2000, high=80034, size=11)
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
    lens = np.random.randint(21000, high=80034, size=13)
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
    samples = 520100
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

def test_fromreader(demo_data):
    """Verifies that a producer from an EDF file reader produces the correct
    subarrays in each iteration."""

    reader = Reader(demo_data)
    # read all samples from this file
    arr = reader.read(0)

    #build a producer and test equality
    chunksize=100231
    pro = producer(reader, chunksize=chunksize, axis=-1)
    starts = range(0, arr.shape[-1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[-1])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[-1] = slice(start, stop)
        assert np.allclose(arr[tuple(slicer)], pro_arr)

def test_from_chReader(demo_data):
    """Test if the produced data from a reader that only reads a subset of
    the channels produces the correct arrays."""

    reader = Reader(demo_data)
    CHS = [1,3]

    #read all channels and then restrict to CHS
    arr = reader.read(0)
    arr = arr[CHS, :]
    
    #set the reader to read only CHS
    reader.channels = CHS

    #build a producer and test equality
    chunksize=10023
    pro = producer(reader, chunksize=chunksize, axis=-1)
    starts = range(0, arr.shape[-1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[-1])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[-1] = slice(start, stop)
        assert np.allclose(arr[tuple(slicer)], pro_arr)

def test_frommaskedreader(demo_data):
    """Verify that a producer from an EDF file reader with a mask produces
    the correct subarrays in each iteration."""

    reader = Reader(demo_data)
    # read 600k samples
    arr = reader.read(0, 600000)

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

def test_subread_shape(demo_data):
    """Validate that the shape of data produced between random starts & stops
    matches the expected shape for 1000 random reads."""

    reader = Reader(demo_data)
    starts = np.random.randint(low=0, high=reader.shape[-1]-1, size=1000)
    stops = starts + np.random.randint(low=1, high=int(10e6), size=1000)
    # ensure largest stop is within data
    stops = np.minimum(stops, reader.shape[-1])
    for start, stop in zip(starts, stops):
        pro = producer(reader, chunksize=30e5, axis=-1, start=start, stop=stop)
        assert pro.shape[-1] == stop - start
    
    reader.close()

def test_subread_values(demo_data):
    """Validate that the values produced between random starts & stops match the
    expected values for 300 random reads."""

    reader = Reader(demo_data)
    starts = np.random.randint(low=0, high=reader.shape[-1]-1, size=300)
    stops = starts + np.random.randint(low=1, high=int(1e6), size=300)
    # ensure largest stop is within data
    stops = np.minimum(stops, reader.shape[-1])
    for start, stop in zip(starts, stops):

        pro = producer(reader, chunksize=30e5, axis=-1, start=start, stop=stop)
        assert np.allclose(pro.to_array(), reader.read(start, stop))
    
    reader.close()

def test_subread_channels(demo_data):
    """Validate that values produced between random starts & stops for
    a restricted set of channes match the expected values for 300 random
    reads."""

    reader = Reader(demo_data)
    reader.channels = [0, 2]
    starts = np.random.randint(low=0, high=reader.shape[-1]-1, size=300)
    stops = starts + np.random.randint(low=1, high=int(1e6), size=300)
    # ensure largest stop is within data
    stops = np.minimum(stops, reader.shape[-1])
    for start, stop in zip(starts, stops):

        pro = producer(reader, chunksize=30e5, axis=-1, start=start, stop=stop)
        assert np.allclose(pro.to_array(), reader.read(start, stop))
    
    reader.close()

def test_subread_mask(demo_data):
    """Validate that values produced between random starts & stops with a mask
    match the expected values for 300 random reads."""

    reader = Reader(demo_data)
    starts = np.random.randint(low=0, high=reader.shape[-1]-1, size=300)
    stops = starts + np.random.randint(low=1, high=int(1e6), size=300)
    # ensure largest stop is within data
    stops = np.minimum(stops, reader.shape[-1])
    for start, stop in zip(starts, stops):

        mask = np.random.choice([True, False], size=stop-start, p=[.4, .6])
        x = reader.read(start, stop)[:, mask]
        pro = producer(reader, 30e5, axis=-1, start=start, stop=stop, mask=mask)
        assert np.allclose(pro.to_array(), x)
    
    reader.close()
