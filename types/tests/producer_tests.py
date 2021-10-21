import pytest
import numpy as np
from itertools import zip_longest

from openseize.types.producer import producer, MaskedProducer

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

    chunksize=10000
    np.random.seed(9634)
    #make sure to use subarrays of varying lens along chunking axis
    lens = np.random.randint(2000, high=80034, size=50)
    #keep the arrays for comparison and make a generator func
    arrs = [np.random.random((17, l)) for l in lens]
    def g():
        for arr in arrs:
            yield arr
    #create a producer from the generator func g
    pro = producer(g, chunksize=chunksize, axis=-1)
    #fetch and store the arrays from the producer
    pro_arrays = [arr for arr in pro]
    #create arrays for comparison from original and from producer
    arr = np.concatenate(arrs, axis=-1)
    pro_arr = np.concatenate(pro_arrays, axis=-1)
    assert np.allclose(arr, pro_arr)

def test_gen2():
    """Test if producer produces correct subarrays when chunksize exactly
    equals the generator's segment sizes along axis."""

    def g(chs=4, n=50000, segsize=9000, seed=0):
        """A generator of constant sized arrays of segsize along axis=-1."""

        np.random.seed(seed)
        starts = range(0, n, segsize)
        segments = zip_longest(starts, starts[1:], fillvalue=n)
        for start, stop in segments:
            arr = np.random.random((chs, stop-start))
            yield arr

    pro = producer(g, chunksize=9000, axis=-1)
    arr = np.concatenate([x for x in pro], axis=-1)
    probe = np.concatenate([arr for arr in g(seed=0)], axis=-1)
    assert np.allclose(probe, arr)

def test_arr_reverse():
    """Test if a reversed producer built from an array produces correct
    subarrays."""

    chunksize = 1000
    arr = np.random.random((157, 35000))
    pro = producer(arr, chunksize=chunksize, axis=-1)
    rev_pro = reversed(pro)
    rev = np.concatenate([r for r in rev_pro], axis=-1)
    probe = np.flip(arr, axis=-1)
    assert np.allclose(probe, rev)

def test_gen_reverse():
    """Test if a reversed producer built from a generator produces correct
    subarrays."""

    chunksize=4399
    np.random.seed(9631)
    #make sure to use subarrays of varying lens along chunking axis
    lens = np.random.randint(2000, high=80000, size=20)
    #keep the arrays for comparison and make a 1-time generator
    arrs = [np.random.random((12, 10, l)) for l in lens]
    def g():
        for arr in arrs:
            yield arr
    #create a producer from the generator
    pro = producer(g, chunksize=chunksize, axis=-1)
    rev_gen = reversed(pro)
    rev = np.concatenate([arr for arr in rev_gen], axis=-1)
    probe = np.concatenate([arr for arr in arrs], axis=-1)
    probe = np.flip(probe, axis=-1)
    assert np.allclose(probe, rev)

def test_gen_reverse2():
    """Test if a reversed producer yields arrays of the correct shape when
    the chunksize exactly equals the generator segment size. """

    chunksize = 10000
    np.random.seed(0)
    arrs = [np.random.random((2, 10000)) for _ in range(5)]
    def g():
        for arr in arrs:
            yield arr
    pro = producer(g, chunksize, axis=-1)
    rev_gen = reversed(pro)
    rev = np.concatenate([arr for arr in rev_gen], axis=-1)
    probe = np.concatenate([arr for arr in arrs], axis=-1)
    probe = np.flip(probe, axis=-1)
    assert np.allclose(probe, rev)

def test_masked_arr():
    """Tests if a producer of masked arrays yields correct subarrays."""

    chunksize = 10028
    size = 510023
    arr = np.random.random((7, 119, 510023))
    pro = producer(arr, chunksize=chunksize, axis=-1)
    mask = np.random.choice([True, False], size=size)
    mpro = MaskedProducer(pro, mask, chunksize=chunksize, axis=-1)
    result0 = np.concatenate([x for x in mpro], axis=-1)
    result1 = np.take(arr, np.flatnonzero(mask), axis=-1)
    assert np.allclose(result0, result1)

def test_revmask_arr():
    """Tests if a reversed masked producer yields correct subarrays."""

    chunksize = 70012
    size = 235780
    arr = np.random.random((7, size, 52))
    pro = producer(arr, chunksize=chunksize, axis=1)
    mask = np.random.choice([True, False], size=size)
    mpro = MaskedProducer(pro, mask, chunksize, axis=1)
    rev_mask_pro = reversed(mpro)
    rev = np.concatenate([x for x in rev_mask_pro], axis=1)
    probe = np.take(arr, np.flatnonzero(mask), axis=1)
    probe = np.flip(probe, axis=1)
    assert np.allclose(rev, probe)

def test_masked_gen():
    """Test if a producer of masked arrays from a generator yields correct
    subarray shapes."""

    chunksize=10011
    np.random.seed(9634)
    #make sure to use subarrays of varying lens along chunking axis
    lens = np.random.randint(2000, high=80034, size=50)
    #keep the arrays for comparison and make a generator func
    arrs = [np.random.random((17, l)) for l in lens]
    def g():
        for x in arrs:
            yield x
    #create a producer from the generator func g
    pro = producer(g, chunksize=chunksize, axis=-1)
    mask = np.random.choice([True, False], size=sum(lens))
    mpro = MaskedProducer(pro, mask, chunksize, axis=-1)
    full_arr = np.concatenate(arrs, axis=-1)
    for arr, marr  in zip(pro, producer(mask, chunksize, axis=0)):
        print(arr.shape, marr.shape)
    return mask, full_arr


if __name__ == '__main__':

    mask, arr = test_masked_gen()


