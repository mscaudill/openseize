"""

"""

import pytest
import numpy as np
from itertools import zip_longest

from openseize.core.producer import producer, as_producer

def test_fromarray():
    """Verify that a producer built from an array produces the correct
    subarrays on each iteration."""

    #build random data array and producer
    arr = np.random.random((119, 510023))
    chunksize = 2031
    pro = producer(arr, chunksize=chunksize, axis=-1)
    #compare each produced array against sliced out subarray
    starts = range(0, arr.shape[-1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[-1])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[-1] = slice(start, stop)
        assert np.allclose(arr[tuple(slicer)], pro_arr)

def test_fromsequence():
    """Verify that a producer built from a sequence of arrays produces the
    correct subarrays on each iteration."""

    # build a sequence of random arrays that vary in samples along 1st axis
    arrs = [np.random.random((3, x, 4)) 
            for x in np.random.randint(1000, 100000, size=16)]
    #build full array for easy slicing concatenating along sample axis=1
    arr  = np.concatenate(arrs, axis=1)
    #build producer from the sequence of ndarrays
    chunksize = 780
    pro = producer(arrs, chunksize=chunksize, axis=1)
    #test equality
    starts = range(0, arr.shape[1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[1])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[1] = slice(start, stop)
        assert np.allclose(arr[tuple(slicer)], pro_arr)

def test_fromgenerator():
    """Verify that a producer built from a generator function yielding
    subarrays produces the correct subarrays on each iteration."""

    #build random test arrays and concatenate for testing
    lens = np.random.randint(2000, high=80034, size=50)
    arrs = [np.random.random((l, 2, 17)) for l in lens]
    arr = np.concatenate(arrs, axis=0)

    def g(arrs):
        """Generating function of random arrays of varying lengths along
        0th axis."""

        for arr in arrs:
            yield arr

    chunksize=10000
    #create a producer from the generator func g passing in arrs as kwargs
    pro = producer(g, chunksize=chunksize, axis=0, shape=arr.shape[0],
                   arrs=arrs)
    #test equality
    starts = range(0, arr.shape[0], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[0])
    for (start, stop), pro_arr in zip(segments, pro):
        slicer = [slice(None)] * arr.ndim #slice obj to slice arr
        slicer[0] = slice(start, stop)
        assert np.allclose(arr[tuple(slicer)], pro_arr)

def test_fromgenedge():
    """Verify that a producer from a generator yields the correct subarrays
    when the chunksize and generator size equal."""

    size = 6000

    def g(chs=4, samples=500000, segsize=size):
        """A generator of constant sized arrays of segsize along axis=-1.

        chs: int
            The size of each yielded array on 0th axis.
        samples: int 
            The total size of all yielded arrays along 1st axis.
        segsize:
            The number of samples along 1st axis yielded  per iteration of
            this generating function.
        """

        rng = np.random.default_rng(seed=21)
        starts = range(0, samples, segsize)
        segments = zip_longest(starts, starts[1:], fillvalue=samples)
        for start, stop in segments:
            arr = rng.random((chs, stop-start))
            yield arr
    
    #build a producer
    pro = producer(g, chunksize=size, axis=-1, shape=(4,500000))
    #test equality
    for idx, (gen_arr, pro_arr) in enumerate(zip(g(), pro)):
        assert np.allclose(gen_arr, pro_arr)

if __name__ == '__main__':

    test_fromgenedge()

