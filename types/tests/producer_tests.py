import pytest
import numpy as np
from itertools import zip_longest

from openseize.types.producer import producer

######################## ARRAY TESTS ##############################

def test_arr0():
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
        assert np.allclose(arr[tuple(slicer)], sub_arr)

def test_arr_reverse0():
    """Test if a reversed producer built from an array produces correct
    subarrays."""

    chunksize = 1000
    arr = np.random.random((157, 35000))
    pro = producer(arr, chunksize=chunksize, axis=-1)
    rev_pro = reversed(pro)
    rev = np.concatenate([r for r in rev_pro], axis=-1)
    probe = np.flip(arr, axis=-1)
    assert np.allclose(probe, rev)

def test_arr_reverse1():
    """Test that a reversed producer yields the expected arrays."""

    chunksize = 2034
    size = 39451
    arr = np.random.random((7, 22, size))
    x = np.flip(arr, axis=-1)
    #build a producer and reverse it
    pro = producer(arr, chunksize, axis=-1)
    revpro = reversed(pro)
    rev_gen = iter(revpro)
    #test each yielded subarr equality
    starts = range(0, arr.shape[-1], chunksize)
    segments = zip_longest(starts, starts[1:], fillvalue=arr.shape[-1])
    for idx, (a,b) in enumerate(segments):
        subarr = next(rev_gen)
        probe = np.take(x, range(a,b), axis=-1)
        assert np.allclose(probe, subarr)

##################### GENERATOR TESTS ##############################

def test_gen0():
    """Test if concatenated arrays from gen producer match test array."""

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

def test_gen1():
    """Test if concatenated arrays from producer match test array when the
    chunksize equals the generators subarr size."""

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

def test_gen2():
    """Test if a producer of masked arrays from a generator yields correct
        subarray shapes."""

    chunksize=10011
    np.random.seed(9634)
    #make subarrays of varying lens along chunking axis
    lens = np.random.randint(2000, high=80034, size=50)
    arrs = [np.random.random((17, l)) for l in lens]

    def g():
        """A generator yielding arrays of varying size."""
        for x in arrs:
            yield x

    #create a producer from the generator func g
    pro = producer(g, chunksize=chunksize, axis=-1)
    p = iter(pro)
    #create the 'ground-truth' array
    arr = np.concatenate(arrs, axis=-1)
    #check that shape of each yielded array is correct
    starts = range(0, arr.shape[-1], chunksize)
    for idx, (start, stop) in enumerate(zip_longest(starts, starts[1:],
                                        fillvalue=arr.shape[-1])):
        probe = np.take(arr, np.arange(start, stop), axis=-1)
        subarr = next(p)
        assert np.allclose(probe.shape, subarr.shape)

def test_gen3():
    """Asserts that the yielded subarr values from a generator match slices
    of a test array."""

    chunksize=10231
    np.random.seed(999)
    #make subarrays of varying lens along chunking axis
    lens = np.random.randint(2000, high=91034, size=50)
    arrs = [np.random.random((2, 17, l)) for l in lens]

    def g():
        """A generator yielding arrays of varying size."""
        for x in arrs:
            yield x

    #create a producer from the generator func g
    pro = producer(g, chunksize=chunksize, axis=-1)
    p = iter(pro)
    #create the 'ground-truth' array
    arr = np.concatenate(arrs, axis=-1)
    #check that shape of each yielded array is correct
    starts = range(0, arr.shape[-1], chunksize)
    for idx, (start, stop) in enumerate(zip_longest(starts, starts[1:],
                                        fillvalue=arr.shape[-1])):
        probe = np.take(arr, np.arange(start, stop), axis=-1)
        subarr = next(p)
        assert np.allclose(probe, subarr)

def test_revgen0():
    """Test if concatenated arrays from a reversed gen producer match the
    full arr."""
    
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

def test_revgen1():
    """Test if a reversed producer yields arrays of the correct shape when
    the chunksize exactly equals the generator segment size. """

    chunksize = 10000
    np.random.seed(0)
    arrs = [np.random.random((2, 10000)) for _ in range(15)]
    def g():
        for arr in arrs:
            yield arr
    pro = producer(g, chunksize, axis=-1)
    rev_gen = reversed(pro)
    rev = np.concatenate([arr for arr in rev_gen], axis=-1)
    probe = np.concatenate([arr for arr in arrs], axis=-1)
    probe = np.flip(probe, axis=-1)
    assert np.allclose(probe, rev)

def test_revgen2():
    """Test if arrays from a reversed producer match expected subarrs."""

    chunksize=20227
    np.random.seed(999)
    #make subarrays of varying lens along chunking axis
    lens = np.random.randint(2000, high=91034, size=50)
    arrs = [np.random.random((3, l, 9)) for l in lens]
    arr = np.concatenate(arrs, axis=1)
    #reverse the array
    rev_arr = np.flip(arr, axis=1)

    def g():
        """A generator yielding arrays of varying size."""
        for x in arrs:
            yield x

    #make a producer and reverse it
    pro = producer(g, chunksize=chunksize, axis=1)
    rev_gen = reversed(pro)
    starts = range(0, arr.shape[1], chunksize)
    for idx, (start, stop) in enumerate(zip_longest(starts, starts[1:],
                                        fillvalue=arr.shape[1])):
        probe = np.take(rev_arr, np.arange(start, stop), axis=1)
        subarr = next(rev_gen)
        assert np.allclose(probe, subarr)

###################### MASK TESTS #######################################

def test_masked_arr0():
    """Tests if subarrs from a producer equals the original arr."""

    chunksize = 10028
    size = 510023
    arr = np.random.random((7, 119, size))
    pro = producer(arr, chunksize=chunksize, axis=-1)
    mask = np.random.choice([True, False], size=size)
    mpro = producer(pro, chunksize=chunksize, axis=-1, mask=mask)
    result0 = np.concatenate([x for x in mpro], axis=-1)
    result1 = np.take(arr, np.flatnonzero(mask), axis=-1)
    assert np.allclose(result0, result1)

def test_masked_arr1():
    """Test if each subarr from a masked producer equals expected subarr."""

    chunksize = 70028
    size = 4967882
    arr = np.random.random((3, 8, size))
    pro = producer(arr, chunksize=chunksize, axis=-1)
    mask = np.random.choice([True, False], size=size)
    mpro = producer(pro, chunksize=chunksize, axis=-1, mask=mask)
    mp = iter(mpro)
    starts = range(0, arr.shape[-1], chunksize)
    for idx, (start, stop) in enumerate(zip_longest(starts, starts[1:],
                                        fillvalue=arr.shape[-1])):
        probe = np.take(arr, np.arange(start, stop), axis=-1)
        filt = np.take(mask, np.arange(start, stop))
        probe = np.take(probe, np.flatnonzero(filt), axis=-1)
        x = next(mp)
        assert np.allclose(probe, x)

def test_revmasked_arr0():
    """Test if subarrs from a rev producer == the reverse of orig. arr."""

    chunksize = 109831
    size = 4967882
    arr = np.random.random((8, size))
    pro = producer(arr, chunksize=chunksize, axis=-1)
    mask = np.random.choice([True, False], size=size)
    mpro = producer(pro, chunksize=chunksize, axis=-1, mask=mask)
    #reverse the masked producer
    mgen = reversed(mpro)
    result0 = np.concatenate([x for x in mgen], axis=-1)
    result1 = np.flip(np.take(arr, np.flatnonzero(mask), axis=-1), axis=-1)
    assert np.allclose(result0, result1)

def test_revmasked_arr1():
    """Test if each subarr from a reversed masked pro matches expected
    subarr."""

    np.random.seed(9634)
    chunksize = 90123
    size = 4967882
    arr = np.random.random((8, size))
    flipped_arr = np.flip(arr, axis=-1)
    mask = np.random.choice([True, False], size=size)
    flipped_mask = np.flip(mask, axis=-1)
    mpro = producer(arr, chunksize=chunksize, axis=-1, mask=mask)
    #reverse the masked producer
    mrgen = reversed(mpro)
    starts = range(0, arr.shape[-1], chunksize)
    for idx, (start, stop) in enumerate(zip_longest(starts, starts[1:],
                                        fillvalue=arr.shape[-1])):
        probe = np.take(flipped_arr, np.arange(start, stop), axis=-1)
        filt = np.take(flipped_mask, np.arange(start, stop), axis=-1)
        y = np.take(probe, np.flatnonzero(filt), axis=-1)
        x = next(mrgen)
        assert np.allclose(y, x)

def test_masked_gen0():
    """Test if a producer of masked arrays from a generator yields correct
    concatenated subarrays."""

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
    mpro = producer(pro, chunksize, axis=-1, mask=mask)
    full_arr = np.concatenate(arrs, axis=-1)
    probe = np.take(full_arr, np.flatnonzero(mask), axis=-1)
    test_arr = np.concatenate([x for x in mpro], axis=-1)
    assert np.allclose(probe, test_arr)

def test_masked_gen1():
    """Test if a producer of masked arrays from a generator yields correct
    subarrays."""

    chunksize=30210
    np.random.seed(7)
    #make sure to use subarrays of varying lens along chunking axis
    lens = np.random.randint(2000, high=80034, size=50)
    #keep the arrays for comparison and make a generator func
    arrs = [np.random.random((6, 13, l)) for l in lens]
    def g():
        for x in arrs:
            yield x
    #create a producer from the generator func g
    pro = producer(g, chunksize=chunksize, axis=-1)
    mask = np.random.choice([True, False], size=sum(lens))
    mpro = producer(pro, chunksize, axis=-1, mask=mask)
    mgen = iter(mpro)
    full_arr = np.concatenate(arrs, axis=-1)
    starts = range(0, full_arr.shape[-1], chunksize)
    for (a, b) in zip_longest(starts, starts[1:], 
                              fillvalue=full_arr.shape[-1]):
        probe = np.take(full_arr, range(a, b), axis=-1)
        filt = np.take(mask, range(a,b), axis=-1)
        probe = np.take(probe, np.flatnonzero(filt), axis=-1)
        subarr = next(mgen)
        assert np.allclose(probe, subarr)

def test_revmasked_gen1():
    """Test if a reverse producer of masked arrays from a generator yields
    correct subarrays."""

    chunksize=18930
    np.random.seed(7)
    #make sure to use subarrays of varying lens along chunking axis
    lens = np.random.randint(2000, high=80034, size=50)
    #keep the arrays for comparison and make a generator func
    arrs = [np.random.random((l, 13, 6)) for l in lens]
    def g():
        for x in arrs:
            yield x
    #create a producer from the generator func g
    mask = np.random.choice([True, False], size=sum(lens))
    mpro = producer(g, chunksize, axis=0, mask=mask)
    m_revgen = iter(reversed(mpro))
    flipped_arr = np.flip(np.concatenate(arrs, axis=0), axis=0)
    flipped_mask = np.flip(mask)
    starts = range(0, flipped_arr.shape[0], chunksize)
    for (a, b) in zip_longest(starts, starts[1:], 
                              fillvalue=flipped_arr.shape[0]):
        probe = np.take(flipped_arr, range(a, b), axis=0)
        filt = np.take(flipped_mask, range(a,b), axis=0)
        probe = np.take(probe, np.flatnonzero(filt), axis=0)
        subarr = next(m_revgen)
        assert np.allclose(probe, subarr)




