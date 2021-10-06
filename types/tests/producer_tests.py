import pytest
import numpy as np
from itertools import zip_longest

from openseize.types.producer import producer

def test_shape():
    """ """

    pass

def test_array():
    """Test if producer produces correct subarrays from a supplied array."""
    
    chunksize = 10028
    arr = np.random.random((119, 510023))
    pro = producer(arr, chunksize=chunksize, axis=-1)
    #compute start, stop segments to extract from arr and compare
    starts = range(0, arr.shape[-1], chunksize)
    segs = zip_longest(starts, starts[1:], fillvalue=arr.shape[-1])
    for seg, sub_arr in zip(segs, pro):
        #place seg slice into nd slicer
        slicer = [slice(None)] * arr.ndim
        slicer[-1] = slice(*seg)
        #compare
        result = np.allclose(arr[tuple(slicer)], sub_arr)
    assert np.all(result)

def test_gen():
    """Test if producer produces correct subarrays from a generator."""

    chunksize=2231
    np.random.seed(9634)
    #make sure to use subarrays of varying lens along chunking axis
    lens = np.random.randint(2000, high=40000, size=50)
    #keep the arrays for comparison and make a 1-time generator
    arrs = [np.random.random((17, l)) for l in lens]
    gen = (arr for arr in arrs)
    #create a producer from the generator
    pro = producer(gen, chunksize=chunksize, axis=-1)
    #fetch and store the arrays from the producer
    pro_arrays = [arr for arr in pro]
    #create arrays for comparison from original and from producer
    arr = np.concatenate(arrs, axis=-1)
    pro_arr = np.concatenate(pro_arrays, axis=-1)
    assert np.allclose(arr, pro_arr)

