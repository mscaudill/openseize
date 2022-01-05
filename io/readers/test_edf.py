import pytest
import numpy as np

from openseize.io import edf
from scripting.spectrum.io.eeg import EEG as OEEG
from scripting.spectrum.io.readers.readers import EDF as OReader

"""Call !pytest test_edf.py::<TEST> to run a single test from ipython
interpreter."""

PATH = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
WRITEPATH = '/home/matt/python/nri/data/openseize/test_write.edf'
UPATH = '/home/matt/python/nri/data/openseize/test_uneven.edf'

def write(channels=[0,1,2,3]):
    """Writes a fresh edf from a known edf."""

    with edf.open_edf(WRITEPATH, 'wb') as outfile:
        with edf.open_edf(PATH, 'rb') as supply:
            outfile.write(supply.header, supply, channels=channels)

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
    oreader = OReader(PATH)
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


"""Tests of 'uneven' data with different sample rates for each channel."""

def data(nrecords, samples_per_record, padval, seed):
    """Returns a random 2-D array of data to simulate an edf data 
    with different sample rates.

    Args:
        nrecords (int):             num of records to write
        samples_per_record (list):  seqeunce that defines number of samples
                                    for each channel within each edf record
        padval (float):             padding value used to make rectangular
                                    array. This will pad channels with less
                                    than max(samples_per_record)
        seed (int):                 random seed for reproducibility

    Returns: ndarray of shape len(samples_per_record) x (nrecords
             * max(samples_per_record)
    """

    np.random.seed(seed)
    spr = samples_per_record
    shape = len(spr),  max(spr) * nrecords
    #build random samples for each channel in spr
    arrs = [1000*np.random.random(s * nrecords) for s in spr]
    #pad all arrays to the max array length and stack
    for i, x in enumerate(arrs):
        arrs[i] = np.pad(x, (0, shape[1] - len(x)), constant_values=padval)
    return np.stack(arrs, axis=0)

def uneven_write(nrecs=100, spr=[50000, 50000, 10000, 20000]):
    """Writes simulated data with uneven spr to an edf file for testing."""

    arr = data(nrecs, spr, padval=0, seed=0)
    #use a prebuilt header and alter it to match our data
    header = edf.Reader(PATH).header
    header = header.filter('channels', [0,1,2,3])
    header['num_records'] = nrecs
    header['samples_per_record'] = spr
    #write the header and data to a new path
    with edf.open_edf(UPATH, 'wb') as outfile:
        outfile.write(header, arr, channels=[0,1,2,3])

def test_uneven_randomread(fetches=1000):
    """Compares all channels for a random fetch in fetches."""

    arr = data(100, [50000, 50000, 10000, 20000], padval=0, seed=0)
    reader = edf.Reader(UPATH)
    #build fetch number of starts and stop indices
    starts = np.random.randint(0, 5e6, fetches)
    stops = starts + np.random.randint(0, 5e4)
    #compare reads between read array and simulated arr
    # NOTE there is a loss of precision as edf uses 2-byte ints so tolerance
    # for comparison is set to within 1 unit
    # for this test the data was padded with 0s so supply the same to reader
    for start, stop in zip(starts, stops):
        x = reader.read(start, stop, padvalue=0)
        assert np.allclose(x, arr[:, start:stop], atol=1)
    reader.close()

def test_uneven_gen(chunksize=1000):
    """Test if the read generator yields the same chunks of data as the
    supplying array used to build the edf."""

    arr = data(100, [50000, 50000, 10000, 20000], padval=0, seed=0)
    reader = edf.Reader(UPATH)
    starts = np.arange(0, arr.shape[1], chunksize)
    stops = starts[1:]
    #for this test the data was padded with 0s so supply the same to reader
    for idx, chunk in enumerate(reader.read(0, chunksize=chunksize,
                                padvalue=0)):
        assert(np.allclose(chunk, arr[:, starts[idx]:stops[idx]], atol=1))
    reader.close()

def test_uneven_chrandomread(fetches=1000, channels=[0,3]):
    """Test if reading a subset of the channels matches a subset of the
    channels from the supplying array used to make the edf."""

    arr = data(100, [50000, 50000, 10000, 20000], padval=0, seed=0)
    reader = edf.Reader(UPATH)
    #build fetch number of starts and stop indices
    starts = np.random.randint(0, 5e5, fetches)
    stops = starts + np.random.randint(0, 5e4)
    #compare reads between read array and simulated arr
    # NOTE there is a loss of precision as edf uses 2-byte ints so tolerance
    # for comparison is set to within 1 unit
    for start, stop in zip(starts, stops):
        x = reader.read(start, stop, channels=channels, padvalue=0)
        assert np.allclose(x, arr[channels, start:stop], atol=1)
    reader.close()

def test_uneven_chgen(chunksize=1000, channels=[1,2]):
    """Test if the read gen for a specific set of channels returns the same
    data as the arr for the same channels."""

    arr = data(100, [50000, 50000, 10000, 20000], padval=0, seed=0)
    reader = edf.Reader(UPATH)
    starts = np.arange(0, arr.shape[1], chunksize)
    stops = starts[1:]
    #for this test the data was padded with 0s so supply the same to reader
    for idx, chunk in enumerate(reader.read(0, chunksize=chunksize,
                                channels=channels, padvalue=0)):
        assert(np.allclose(chunk, arr[channels, starts[idx]:stops[idx]], atol=1))
    reader.close()


   


"""
if __name__ == '__main__':

    uneven_write()
"""
