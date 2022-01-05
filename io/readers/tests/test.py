"""A series of tests comparing the openseize EDF reader against the pyedf
reader. The repository for this code can be found here:

https://github.com/bids-standard/pyedf/tree/py3k

Typical usage example:
    !pytest test.py::<TEST_NAME>
"""

import pytest
import numpy as np
from openseize.io.readers.tests.pyedf.EDF import EDFReader as oEDF
from openseize.io.readers.readers import EDF

DATAPATH = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'

def oread(eeg, start, stop, channels=None):
    """Reads using the pyEDF reader between start and stop samples for each
    channel in channels.

    The pyEDF reader (unlike openseize) cannot read multiple channels 
    simultaneously so this funtion performs multiple reads to gather data
    across channels.

    Args:
        eeg: pyEDF.EDFReader instance
            A pyEDF.EDFReader object with a read_samples method.
        start: int
            The start sample index to read.
        stop: int
            The stop sample index to read (exclusive).
        channels: sequence
            A sequence of channels to read using pyEDF Reader. If None,
            reads all the channels in the EDF. Default is None.
    
    Returns:
        A 2-D array of data of shape len(channels) x (stop-start) samples.
    """

    if not channels:
        chs = range(len(eeg.chan_info['ch_names']) - 1) #-1 for annotations
    else:
        chs = channels

    result = []
    for channel in chs:
        #pyedf is inclusive of stop
        result.append(eeg.read_samples(channel, start, stop-1))
    return np.array(result)


def test_random_reads(fetches=1000):
    """Compares randomly read arrays from pyEDF and openseize."""

    oeeg = oEDF(DATAPATH)
    eeg = EDF(DATAPATH)

    #build fetch numbers of start and stop sample indices
    starts = np.random.randint(0, 5e8, fetches)
    stops = starts + np.random.randint(0, 5e5)
    
    for start, stop in zip(starts, stops):
        arr = eeg.read(start, stop)
        other = oread(oeeg, start, stop)
        assert np.allclose(arr, other)

    eeg.close()
    oeeg.close()


def test_random_reads_chs(fetches=1000, channels=[0,2]):
    """Compares randomly read arrays for a specifc set of channels from
    pyEDF and openseize."""

    oeeg = oEDF(DATAPATH)
    eeg = EDF(DATAPATH)

    #build fetch numbers of start and stop sample indices
    starts = np.random.randint(0, 5e8, fetches)
    stops = starts + np.random.randint(0, 5e5)
    
    for start, stop in zip(starts, stops):
        arr = eeg.read(start, stop, channels=channels)
        other = oread(oeeg, start, stop, channels=channels)
        assert np.allclose(arr, other)

    eeg.close()
    oeeg.close()

def test_read_EOF():
    """Test if start sample is at EOF that reader returns an empty array."""

    eeg = EDF(DATAPATH)
    start = max(eeg.header.samples) + 1
    arr = eeg.read(start, start+100)
    assert arr.size == 0


def test_read_EOF2():
    """Test if array from slice on and off end of EDF has correct shape."""

    edf = EDF(DATAPATH)
    #read 200 samples starting from 100 samples before EOF
    start = max(edf.header.samples) - 100
    arr = edf.read(start, start + 200)
    assert arr.shape[-1] == 100







