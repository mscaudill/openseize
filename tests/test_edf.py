"""A module for testing the reading and writing of edf files. The tests are
compared against the pyedf library results. This library can be found here:

https://github.com/bids-standard/pyedf/tree/py3k

Typical usage example:
    !pytest edf_tests.py::<TEST_NAME>
"""

import pytest
import numpy as np
from pathlib import Path
from openseize.file_io.tests.pyedf.EDF import EDFReader as pyEDF
from openseize.file_io.edf import Reader as openEDF
from openseize.file_io.edf import Header as openHeader
from openseize.file_io.edf import Writer as openWriter
from openseize.file_io.edf import splitter
from openseize.demos import paths

WRITEPATH = paths.data_dir.joinpath('write_test.edf')
IRREGULARPATH = paths.data_dir.joinpath('irregular.edf')

################
# HEADER TESTS #
################

def test_header(demo_data):
    """Test if the header values from opensieze match pyedf.

    Unfortunately, pyEDF changes the field namesi and or values from the
    EDF specification when they internally store the data to the EDFREADER 
    so to compare with opensiezes EDF compliant Header, we must compare 
    fields one by one.
    """

    pyeeg = pyEDF(demo_data)
    openheader = openHeader(demo_data)
    assert(openheader.version == pyeeg.meas_info['file_ver'])
    assert(openheader.patient == pyeeg.meas_info['subject_id'])
    assert(openheader.recording == pyeeg.meas_info['recording_id'])
    #dates & times in pyedf are not compliant with EDF specs
    pydate = [str(pyeeg.meas_info[x]) for x in ['day', 'month', 'year']]
    pydate = ['0' + x if len(x) < 2 else x for x in pydate]
    assert(openheader.start_date == '.'.join(pydate))
    pytime = [str(pyeeg.meas_info[x]) for x in 'hour minute second'.split()]
    pytime = ['0' + x if len(x) < 2 else x for x in pytime]
    assert openheader.start_time == '.'.join(pytime)
    assert openheader.header_bytes == pyeeg.meas_info['data_offset']
    # pyedf does not handle reserve section correctly. The 44 bytes of this
    # section hold the type of edf file. pyedf uses the file extension if
    # this is empty in the header but this fails to distinguish edf from
    # edf+. We therefore do not compare this field.
    assert openheader.num_records == pyeeg.meas_info['n_records']
    assert openheader.record_duration == pyeeg.meas_info['record_length']
    assert openheader.num_signals == pyeeg.meas_info['nchan']
    assert openheader.names == pyeeg.chan_info['ch_names']
    assert openheader.transducers == pyeeg.chan_info['transducers']
    assert openheader.physical_dim == pyeeg.chan_info['units']
    assert np.allclose(openheader.physical_min, 
                       pyeeg.chan_info['physical_min'])
    assert np.allclose(openheader.physical_max, 
                       pyeeg.chan_info['physical_max'])
    assert np.allclose(openheader.digital_min,
                       pyeeg.chan_info['digital_min'])
    assert np.allclose(openheader.digital_max, 
                       pyeeg.chan_info['digital_max'])


################
# READER TESTS #
################

def pyread(eeg, start, stop, channels=None):
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


def test_random_reads(demo_data):
    """Compares randomly read arrays from pyEDF and openseize."""

    pyeeg = pyEDF(demo_data)
    openeeg = openEDF(demo_data)

    fetches = 10
    #build fetch numbers of start and stop sample indices
    starts = np.random.randint(0, 5e6, fetches)
    stops = starts + np.random.randint(0, 5e4)
    
    for start, stop in zip(starts, stops):
        arr = openeeg.read(start, stop)
        other = pyread(pyeeg, start, stop)
        assert np.allclose(arr, other)

    openeeg.close()
    pyeeg.close()

def test_random_reads_chs(fetches=10):
    """Compares randomly read arrays for a specifc set of channels from
    pyEDF and openseize."""

    chs = [1,3]
    pyeeg = pyEDF(demo_data)
    openeeg = openEDF(demo_data)
    openeeg.channels = chs

    #build fetch numbers of start and stop sample indices
    starts = np.random.randint(0, 5e6, fetches)
    stops = starts + np.random.randint(0, 5e5)
    
    for start, stop in zip(starts, stops):
        arr = openeeg.read(start, stop)
        other = pyread(pyeeg, start, stop, channels=chs)
        assert np.allclose(arr, other)

    pyeeg.close()
    openeeg.close()

def test_read_EOF():
    """Test if start sample is at EOF that reader returns an empty array."""

    openeeg = openEDF(demo_data)
    start = max(openeeg.header.samples) + 1
    arr = openeeg.read(start, start+100)
    assert arr.size == 0

    openeeg.close()

def test_read_EOF2():
    """Test if array from slice on and off end of EDF has correct shape."""

    openeeg = openEDF(demo_data)
    #read 200 samples starting from 100 samples before EOF
    start = max(openeeg.header.samples) - 100
    arr = openeeg.read(start, start + 200)
    assert arr.shape[-1] == 100

    openeeg.close()

################
# WRITER TESTS #
################

def test_written_header(demo_data, written_data):
    """The written data fixture (see conftest) wrote channels [0, 3] from
    the demo_data edf to a new file 'write_test.edf'. This test asserts that
    the correct header data was written to that file."""

    channels=[0, 3]
    # open to get the unfiltered header
    with openEDF(demo_data) as reader:
        header = reader.header
    
    # open written to get the filtered header
    with openEDF(written_data) as reader:
        filtered_header = reader.header
        
    assert filtered_header == header.filter(channels)


def test_written_data(demo_data, written_data):
    """The written data fixture (see conftest) wrote channels [0, 3] from
    the demo_data edf to a new file 'write_test.edf'. This test asserts that
    the correct data was written to that file."""

    openeeg = openEDF(demo_data)
    openeeg2 = openEDF(written_data)

    #read data in steps of 5 million samples
    starts = np.arange(0, openeeg.shape[-1], int(5e6))
    stops = starts + int(5e3)
    for start, stop in zip(starts, stops):
        arr = openeeg.read(start, stop)
        other = openeeg2.read(start, stop)
        assert np.allclose(arr[[0, 3],:], other)

#################################
# IRREGULAR DATA READ AND WRITE #
#################################

def data(nrecords, samples_per_record, padval, seed):
    """Creates an array of random values to write to an EDF.

    Makes a channels x samples array of random values. Channels
    that run out of samples (due to lower sampling rates) will be filled
    with padvalue.

    Args:
        nrecords: int
            Number of records to be written to test file.
        samples_per_record: sequence
            Number of samples that will be written for each channel to
            a single record. 
        padval: float
            Value to pad to channels that run out of samples due to lower
            sampling rate than other channels.
        seed: int
            Random number seeding for reproducible results.

    Returns: 2-D array
        A len(samples_per_record) x samples_per_record array.
    """

    rng = np.random.default_rng(seed)
    num_chs = len(samples_per_record)
    # create random data array
    x = padval * np.ones((num_chs, max(samples_per_record) * nrecords))
    for ch in range(num_chs):
        stop = samples_per_record[ch] * nrecords
        x[ch, :stop] = 1000 * rng.random(samples_per_record[ch] * nrecords)
    return x


#FIXME: FIXTURE
def irregular_write(nrecords=200, 
                    samples_per_record=[50000, 10000, 20000, 50000], 
                    pad_val=np.NaN, seed=0, path=IRREGULARPATH):
    """Writes an EDF file of random data with different sample rates across
    the channels.

    Please see data for futher details.
    """

    arr = data(nrecords, samples_per_record, pad_val, seed)
    #build a header using a prebuilt header
    header = openEDF(path=DATAPATH).header
    header = header.filter(list(range(len(samples_per_record))))
    header['num_records'] = nrecords
    header['samples_per_record'] = samples_per_record
    #write header and data to irregular path
    chs = list(range(len(samples_per_record)))
    with openWriter(path) as outfile:
        outfile.write(header, arr, channels=chs)

def test_irr_read():
    """Test if irregular data returned by reader matches written irregular
    data."""

    # build the same data used to write and compare against written
    oarr = data(200, [50000, 10000, 20000, 50000], padval=np.NAN, seed=0)
    reader = openEDF(IRREGULARPATH)
    read_arr = reader.read(start=0)
    #imprecision due to 2-byte conversion so tolerance set to 1 unit
    assert np.allclose(oarr, read_arr, equal_nan=True, atol=1)
    reader.close()

def test_irr_randomread(fetches=1000):
    """Test if each random fetch from an irregular EDF matches the written
    EDF."""

    oarr = data(200, [50000, 10000, 20000, 50000], padval=np.NAN, seed=0)
    reader = openEDF(IRREGULARPATH)
    starts = np.random.randint(0, 9e6, fetches)
    stops = starts + np.random.randint(0, 1e6)
    for start, stop in zip(starts, stops):
        x = reader.read(start, stop, padvalue=np.NAN)
        assert np.allclose(x, oarr[:, start:stop], equal_nan=True, atol=1)
    reader.close()

def test_irr_random_chread(fetches=1000, channels=[1,3]):
    """Test if each random fetch from an irregular EDF matches the written
    EDF for a specific set of channels."""

    oarr = data(200, [50000, 10000, 20000, 50000], padval=np.NAN, seed=0)
    oarr = oarr[channels]
    reader = openEDF(IRREGULARPATH)
    reader.channels = channels
    starts = np.random.randint(0, 9e6, fetches)
    stops = starts + np.random.randint(0, 1e6)
    for start, stop in zip(starts, stops):
        x = reader.read(start, stop, padvalue=np.NAN)
        assert np.allclose(x, oarr[:, start:stop], equal_nan=True, atol=1)
    reader.close()

##################
# SPLITTER TESTS #
##################

def test_header_split():
    """Opens an edf, splits the file, and checks that each split header
    contains the correct metadata."""
    
    mapping={'file0':[0,1], 'file1':[2,3]}
    splitter(IRREGULARPATH, mapping=mapping)

    #get unsplit header
    with openEDF(IRREGULARPATH) as infile:
        unsplit_header = infile.header

    #test each split header
    for fname, indices in mapping.items():
        loc = Path(IRREGULARPATH).parent.joinpath(fname).with_suffix('.edf')
        with openEDF(loc) as infile:
            header = infile.header
        probe = unsplit_header.filter(indices)
        assert header == probe

def test_data_split():
    """Opens split files (see test_header_split) and test that the data in
    each split file matches the data in the original unsplit file."""

    
    with openEDF(IRREGULARPATH) as infile:
        unsplit_data = infile.read(start=0)

    mapping={'file0':[0,1], 'file1':[2,3]}
    dirname = Path(IRREGULARPATH).parent
    locs = [dirname.joinpath(f).with_suffix('.edf') for f in mapping]
    for fp, indices in zip(locs, mapping.values()):
        with openEDF(fp) as infile:
            arr = infile.read(start=0)
        assert np.allclose(arr, unsplit_data[indices,:], equal_nan=True)    
   
