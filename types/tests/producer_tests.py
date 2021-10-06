import pytest
import numpy as np
from itertools import zip_longest

from openseize.types.producer import producer


def array_test():
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
    print(np.all(result))

def gen_test():
    """ """

    chunksize=2011
    np.random.seed(0)
    lens = np.random.randint(1000, high=40000, size=3)
    print(lens)
    arrs = [np.random.random((21, l)) for l in lens]
    gen = (arr for arr in arrs)
    pro = producer(gen, chunksize=chunksize, axis=-1)

    pro_arrays = []
    for arr in pro:
        pro_arrays.append(arr)
    arr = np.concatenate(arrs, axis=-1)
    pro_arr = np.concatenate(pro_arrays, axis=-1)
    print(np.allclose(arr, pro_arr))
    print(arr.shape, pro_arr.shape)
    return arr, pro_arr

if __name__ == '__main__':

    arr, pro_arr = gen_test()


