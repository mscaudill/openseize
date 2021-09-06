import pytest
import numpy as np

from openseize.io import edf
from scripting.spectrum.io.eeg import EEG as OEEG
from scripting.spectrum.io.readers.readers import EDF as OReader

"""Call !pytest test_edf.py::<TEST> to run a single test from ipython
interpreter."""

PATH = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
WRITEPATH = '/home/matt/python/nri/data/openseize/test_write.edf'


def test_read_array(fetches=10000):
    """Test if array fetching with the new reader matches arrays returned
    from the old reader."""

    reader = edf.Reader(PATH)
    oreader = OReader(PATH)
    #build fetch number of starts and stop indices
    starts = np.random.randint(0, 5e8, fetches)
    stops = starts + np.random.randint(0, 5e5)
    #compare reads between readers
    for start, stop in zip(starts, stops):
        arr = reader.read(start, stop)
        oarr = oreader.read(start, stop)
        assert np.allclose(arr.T, oarr)
    reader.close()

def test_read_generator():
    """Test if readers generator yields the same arrays as the old EEG
    iterable."""

    reader = edf.Reader(PATH)
    #build the readers generator and compare
    read_gen = reader.read(start=0, chunksize=30e6)
    oeeg = OEEG(PATH, chunksize=30e6)
    for arr, other in zip(read_gen, oeeg):
        assert np.allclose(arr.T, other)
    reader.close()

def test_read_channels(fetches=10000, channels=[2,3]):
    """Test if array fetching with a specified set of channels matches what
    is read by the old EEG iterable."""

    reader = edf.Reader(PATH)
    oreader = OReader(PATH)
    #build fetch number of starts and stop indices
    starts = np.random.randint(0, 5e8, fetches)
    stops = starts + np.random.randint(0, 5e5)
    #compare reads between readers
    for start, stop in zip(starts, stops):
        arr = reader.read(start, stop, channels)
        oarr = oreader.read(start,stop)
        assert np.allclose(arr.T, oarr[:, channels])

def test_readgen_channels(channels=[0,3]):
    """Test if the read generator returns the same same data as the old EEG
    iterable when it reads specific channels."""

    reader = edf.Reader(PATH)
    #build the readers generator and compare
    read_gen = reader.read(start=0, channels=channels, chunksize=30e6)
    oeeg = OEEG(PATH, chunksize=30e6)
    for arr, other in zip(read_gen, oeeg):
        assert np.allclose(arr.T, other[:, channels])
    reader.close()

def test_read_EOF():
    """Test indexing error if start supplied exceeds samples in the file."""

    reader = edf.Reader(PATH)
    #attempt to read 1 sample starting off of file
    start = max(reader.header.samples) + 1
    with pytest.raises(EOFError):
        arr = reader.read(start, start+1)
    reader.close()

def test_read_slice_EOF():
    """Test that slices that read off the end of the file have the correct
    shape."""

    reader = edf.Reader(PATH)
    #read starting at 10 samples prior to EOF and attempt 100 sample read
    start = max(reader.header.samples) - 10
    stop = start + 100
    arr = reader.read(start, stop)
    assert(arr.shape == (len(reader.header.channels), 10))
    reader.close()

def test_read_shape():
    """Test that the readers reported shape is the expected shape."""

    reader = edf.Reader(PATH)
    assert reader.shape == (oreader.num_channels, max(oreader.num_samples))
    reader.close()

def test_write_header_equality():
    """Test if the header of a written EDF matches original header."""

    reader = edf.Reader(PATH)
    #write an EDF from a reader
    with edf.open_edf(WRITEPATH, 'wb') as writer:
        writer.write(reader.header, reader, channels=reader.header.channels)
    #read in the written EDF and test equality
    with edf.open_edf(WRITEPATH, 'rb') as infile:
        header = infile.header
    #filter the original header as writer does not write out annotations
    old_header = reader.header.filter('channels', [0,1,2,3])
    reader.close()
    assert header == old_header

def test_seqwrite():
    """Test if data written sequentially matches the data supplied."""

    supplied = edf.Reader(PATH)
    written = edf.Reader(WRITEPATH)
    for chunk1, chunk2 in zip(supplied.read(0), written.read(0)):
        assert np.allclose(chunk1, chunk2)
    supplied.close()
    written.close()

def test_randomwrite(fetches=10000):
    """Test if randomly selected data written matches the supplied data.

    Note this is more of a test of the reader since seq write works."""

    supplied = edf.Reader(PATH)
    written = edf.Reader(WRITEPATH)
    #build fetch number of starts and stop indices
    starts = np.random.randint(0, 5e8, fetches)
    stops = starts + np.random.randint(0, 5e5)
    for a, b in zip(starts, stops):
        assert np.allclose(supplied.read(a, b), written.read(a, b))
    supplied.close()
    written.close()

def test_channelwrite_header():
    """Writes a file with a specific set of channels and then test for
    equality with supplied file."""

    supplied = edf.Reader(PATH)
    #write out specific channels
    with edf.open_edf(WRITEPATH, 'wb') as outfile:
        outfile.write(supplied.header, supplied, channels=[1,3])
    #filter the input header
    inheader = supplied.header.filter('channels', [1,3])
    supplied.close()
    #read and compare input header with written header
    with edf.open_edf(WRITEPATH, 'rb') as f:
        outheader = f.header
    assert outheader == inheader

def test_channelwrite_randomread(fetches=10000, channels=[1,3]):
    """Test if the data written by test_channelwrite_header for a specified
    set of channels matches what was supplied."""

    supplied = edf.Reader(PATH)
    written = edf.Reader(WRITEPATH)
    #build fetch number of starts and stop indices
    starts = np.random.randint(0, 5e8, fetches)
    stops = starts + np.random.randint(0, 5e5)
    for a, b in zip(starts, stops):
        assert np.allclose(supplied.read(a, b, channels), 
                           written.read(a, b))
    supplied.close()
    written.close()

def test_channelwrite_readgen(channels=[1,3]):
    """Test if reading in the channels written matches what was supplied
    during reader iteration."""

    supplied = edf.Reader(PATH)
    written = edf.Reader(WRITEPATH)
    supplied_gen = supplied.read(start=0, channels=channels)
    written_gen = written.read(start=0)
    for arr, other in zip(written_gen, supplied_gen):
        assert np.allclose(arr, other)
    supplied.close()
    written.close()

"""STILL NEED TO TEST WITH FAKE DATA OF DIFFERENT SAMPLE RATES."""


    
