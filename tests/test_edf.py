"""A module for testing the reading and writing of edf files. The tests are
compared against the pyedf library results. This library can be found here:

https://github.com/bids-standard/pyedf/tree/py3k

Typical usage example:
    !pytest edf_tests.py::<TEST_NAME>
"""

import pytest
import numpy as np
from pathlib import Path
from pyedf.EDF import EDFReader as pyEDF
from openseize.file_io.edf import Reader as openEDF
from openseize.file_io.edf import Header as openHeader
from openseize.file_io.edf import Writer as openWriter
from openseize.file_io.edf import splitter
from openseize.demos import paths


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
    simultaneously so this function performs multiple reads to gather data
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

def test_random_reads_chs(demo_data):
    """Compares randomly read arrays for a specific set of channels from
    pyEDF and openseize."""

    fetches = 10
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

def test_read_EOF(demo_data):
    """Test if start sample is at EOF that reader returns an empty array."""

    openeeg = openEDF(demo_data)
    start = max(openeeg.header.samples) + 1
    arr = openeeg.read(start, start+100)
    assert arr.size == 0

    openeeg.close()

def test_read_EOF2(demo_data):
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

def test_irr_read(irregular_written_data):
    """Test if irregular data returned by reader matches written irregular
    data."""

    fp, written = irregular_written_data
    with openEDF(fp) as reader:
        arr = reader.read(0)
    #imprecision due to 2-byte conversion so tolerance set to 1 unit
    assert np.allclose(written, arr, equal_nan=True, atol=1)

def test_irr_randomread(irregular_written_data):
    """Test if each random fetch from an irregular EDF matches the written
    EDF."""

    fetches = 20
    starts = np.random.randint(0, 9e3, fetches)
    stops = starts + np.random.randint(0, 1e6)

    fp, written = irregular_written_data
    with openEDF(fp) as reader:
        for start, stop in zip(starts, stops):
            arr = reader.read(start, stop, padvalue=np.NaN)
            x = written[:, start:stop]
            assert np.allclose(arr, x, equal_nan=True, atol=1)

def test_irr_random_chread(irregular_written_data):
    """Test if each random fetch from an irregular EDF matches the written
    EDF for a specific set of channels."""

    fetches = 20
    channels = [1,3]
    starts = np.random.randint(0, 5e4, fetches)
    stops = starts + np.random.randint(0, 1e6)

    fp, written = irregular_written_data
    with openEDF(fp) as reader:
        reader.channels = channels
        for start, stop in zip(starts, stops):
            arr = reader.read(start, stop, padvalue=np.NaN)
            x = written[channels, start:stop]
            #imprecision due to 2-byte conversion so tolerance set to 1 unit
            assert np.allclose(arr, x, equal_nan=True, atol=1)

##################
# SPLITTER TESTS #
##################

def test_header_split(irregular_written_data, split_data):
    """Opens an edf, splits the file, and checks that each split header
    contains the correct metadata."""
   
    unsplit_fp, _ = irregular_written_data

    with openEDF(unsplit_fp) as reader:
        unsplit_header = reader.header

    for fp, indices in split_data.items():
        with openEDF(fp) as reader:
            header = reader.header
        assert header == unsplit_header.filter(indices)

def test_data_split(irregular_written_data, split_data):
    """Opens split files (see test_header_split) and test that the data in
    each split file matches the data in the original unsplit file."""

    unsplit_fp, _ = irregular_written_data
    with openEDF(unsplit_fp) as reader:
        unsplit_data = reader.read(start=0)

    for fp, chs in split_data.items():
        with openEDF(fp) as reader:
            arr = reader.read(start=0)

        # since sample rates differ we go to max shape
        nsamples = arr.shape[-1]
        assert np.allclose(arr, unsplit_data[chs, :nsamples], equal_nan=True)    
