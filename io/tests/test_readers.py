import pytest
import numpy as np
from itertools import zip_longest
from openseize.io.readers import EDF
from scripting.spectrum.io.readers.readers import EDF as oEDF
from scripting.spectrum.io.eeg import EEG

"""
# call single test from this file
>>> !pytest test_readers.py::<TEST NAME>
"""

PATH = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'

def test_random_read(fetches=1000):
    """Compares fetch number of random reads from EDF & oEDF."""

    edf = EDF(PATH)
    oedf = oEDF(PATH)
    #build fetch number of starts and stop indices
    starts = np.random.randint(0, 5e8, fetches)
    stops = starts + np.random.randint(0, 5e5)
    #compare reads
    for start, stop in zip(starts, stops):
        arr = edf.read(start, stop)
        other = oedf.read(start, stop)
        assert np.allclose(arr, other.T)
    edf.close()

def test_ch_read(fetches=1000, channels=[1,3]):
    """Compares fetch number of random reads for a subset of channels for
    EDF and oEDF."""

    edf = EDF(PATH)
    oedf = oEDF(PATH)
    #build fetch number of starts and stop indices
    starts = np.random.randint(0, 5e8, fetches)
    stops = starts + np.random.randint(0, 5e5)
    #compare reads
    for start, stop in zip(starts, stops):
        arr = edf.read(start, stop, channels=channels)
        other = oedf.read(start, stop)
        assert np.allclose(arr, other.T[channels,:])
    edf.close()

def test_read_iter():
    """Compares the iterated arrays from EDF with EEG."""
    
    chunksize=30e6
    pro = EDF(PATH).as_producer(chunksize)
    eeg = EEG(PATH, chunksize=chunksize)
    for arr, other in zip(pro, eeg):
        assert np.allclose(arr, other.T)

def test_read_ch_iter(channels=[0,1]):
    """Compares iterated arrays for subset of channels from EDF with EEG."""

    chunksize=45e6
    pro = EDF(PATH).as_producer(chunksize, channels=channels)
    eeg = EEG(PATH, chunksize=chunksize)
    for arr, other in zip(pro, eeg):
        assert np.allclose(arr, other.T[channels,:])

def test_read_EOF():
    """Test if empty array is returned if start sample > len(EDF)."""

    edf = EDF(PATH)
    #attempt to read 10 samples starting off of file
    start = max(edf.header.samples)
    arr = edf.read(start, start + 10)
    assert np.size(arr) == 0

def test_read_EOF2():
    """Test if array from slice on and off end of EDF has correct shape."""

    edf = EDF(PATH)
    #read 200 samples starting from 100 samples before EOF
    start = max(edf.header.samples) - 100
    arr = edf.read(start, start + 200)
    assert arr.shape[-1] == 100

def test_read_reversed():
    """Compares the reversed arrays from EDF with EEG."""

    chunksize=int(30e6)
    edf = EDF(PATH)
    oedf = oEDF(PATH)
    pro = edf.as_producer(chunksize)
    rev = reversed(pro)
    starts = range(edf.shape[-1], 0, -chunksize)
    stops = range(edf.shape[-1] - chunksize, 0, -chunksize)
    for idx, (start, stop) in enumerate(zip_longest(starts, stops,
                                        fillvalue=0)):
        print(idx, start, stop)
        arr = next(rev)
        other = np.flip(oedf.read(stop, start).T, axis=-1)
        print(np.allclose(arr, other))

if __name__ == '__main__':

    test_read_reversed()


