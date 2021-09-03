import pytest
import numpy as np

from openseize.io import edf
from scripting.spectrum.io.eeg import EEG as OEEG
from scripting.spectrum.io.readers.readers import EDF as OReader

"""Call !pytest test_edf.py::<TEST> to run a single test from ipython
interpreter."""

PATH = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
WRITEPATH = '/home/matt/python/nri/data/openseize/test_write.edf'

reader = edf.Reader(PATH)
oreader = OReader(PATH)

def test_read_array(fetches=10000):
    """Test if array fetching with the new reader matches arrays returned
    from the old reader."""

    #build fetch number of starts and stop indices
    starts = np.random.randint(0, 5e8, fetches)
    stops = starts + np.random.randint(0, 5e5)
    #compare reads between readers
    for start, stop in zip(starts, stops):
        arr = reader.read(start, stop)
        oarr = oreader.read(start, stop)
        assert np.allclose(arr.T, oarr)

def test_read_generator():
    """Test if readers generator yields the same arrays as the old EEG
    iterable."""

    #build the readers generator and compare
    read_gen = reader.read(start=0, chunksize=30e6)
    oeeg = OEEG(PATH, chunksize=30e6)
    for arr, other in zip(read_gen, oeeg):
        assert np.allclose(arr.T, other)

def test_read_EOF():
    """Test indexing error if start supplied exceeds samples in the file."""

    #attempt to read 1 sample starting off of file
    start = max(reader.header.samples) + 1
    with pytest.raises(EOFError):
        arr = reader.read(start, start+1)

def test_read_slice_EOF():
    """Test that slices that read off the end of the file have the correct
    shape."""

    #read starting at 10 samples prior to EOF and attempt 100 sample read
    start = max(reader.header.samples) - 10
    stop = start + 100
    arr = reader.read(start, stop)
    assert(arr.shape == (len(reader.header.channels), 10))

def test_write_header_equality():
    """ """

    with edf.open_edf(WRITEPATH, 'wb') as writer:
        pass


    
