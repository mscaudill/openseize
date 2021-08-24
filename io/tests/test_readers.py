import pytest

import numpy as np

from openseize.readers import EDFReader
from scripting.spectrum.io.readers.readers import EDF as OldReader

PATH = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
READER = EDFReader(PATH)
OLDREADER = OldReader(PATH)

def test_equality():
    """Test if the the EDF reader values match the OldReader."""
    
    fetches = 10000
    starts = np.random.randint(0, 200e6, fetches)
    stops = starts + np.random.randint(0, 50e4)
    for start, stop in zip(starts, stops):
        arr = READER.read(start, stop).T
        other = OLDREADER.read(start, stop)
        assert np.allclose(arr, other)

def test_EOF():
    """Test if the last records are equal.

    The old reader actually has 1 more record of appended 0s and will throw
    and error if the stop exceeds the number of samples in the EDF
    """

    start = 1356700000
    stop = start + 50000
    arr = READER.read(start, stop)
    other = OLDREADER.read(start, stop)
    assert np.allclose(arr.T, other)

def test_EOF2():
    """Verifies that the last record read when the stop exceeds the total
    number of samples is equal to the Old readers next to last record."""

    #stop exceeds number of samples in the test file
    start = 1356700000
    stop = start + 100000
    arr = READER.read(start, stop)
    other = OLDREADER.read(start=1356700000, stop=1356750000)
    assert np.allclose(arr.T, other)

def test_EOF3():
    """Test if client request a start > than signal lengths in EDF."""

    #start exceeds number of samples in the test file
    start = 1356900000
    stop = start + 100000
    with pytest.raises(IndexError):
        arr = READER.read(start, stop)


