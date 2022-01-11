"""A module for testing the reading and writing of edf files. The tests are
compared against the pyedf library results. This library can be found here:

https://github.com/bids-standard/pyedf/tree/py3k

Typical usage example:
    !pytest edf_tests.py::<TEST_NAME>
"""

import pytest
import numpy as np
from openseize.io.tests.pyedf.EDF import EDFReader as pyEDF
from openseize.io.edf import Reader as openEDF
from openseize.io.edf import Header as openHeader
from openseize.io.edf import Writer as openWriter

DATAPATH = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
WRITEPATH = '/home/matt/python/nri/data/openseize/test.edf'

################
# HEADER TESTS #
################

def test_header():
    """Test if the header values from opensieze match pyedf.

    Unfortunately, pyEDF changes the field namesi and or values from the
    EDF specification when they internally store the data to the EDFREADER 
    so to compare with opensiezes EDF compliant Header, we must compare 
    fields one by one.
    """

    pyeeg = pyEDF(DATAPATH)
    openheader = openHeader(DATAPATH)
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


def test_random_reads(fetches=100):
    """Compares randomly read arrays from pyEDF and openseize."""

    pyeeg = pyEDF(DATAPATH)
    openeeg = openEDF(DATAPATH)

    #build fetch numbers of start and stop sample indices
    starts = np.random.randint(0, 5e8, fetches)
    stops = starts + np.random.randint(0, 5e5)
    
    for start, stop in zip(starts, stops):
        arr = openeeg.read(start, stop)
        other = pyread(pyeeg, start, stop)
        assert np.allclose(arr, other)

    openeeg.close()
    pyeeg.close()

def test_random_reads_chs(fetches=100, channels=[0,2]):
    """Compares randomly read arrays for a specifc set of channels from
    pyEDF and openseize."""

    pyeeg = pyEDF(DATAPATH)
    openeeg = openEDF(DATAPATH)

    #build fetch numbers of start and stop sample indices
    starts = np.random.randint(0, 5e8, fetches)
    stops = starts + np.random.randint(0, 5e5)
    
    for start, stop in zip(starts, stops):
        arr = openeeg.read(start, stop, channels=channels)
        other = pyread(pyeeg, start, stop, channels=channels)
        assert np.allclose(arr, other)

    pyeeg.close()
    openeeg.close()

def test_read_EOF():
    """Test if start sample is at EOF that reader returns an empty array."""

    openeeg = openEDF(DATAPATH)
    start = max(openeeg.header.samples) + 1
    arr = openeeg.read(start, start+100)
    assert arr.size == 0

    openeeg.close()

def test_read_EOF2():
    """Test if array from slice on and off end of EDF has correct shape."""

    openeeg = openEDF(DATAPATH)
    #read 200 samples starting from 100 samples before EOF
    start = max(openeeg.header.samples) - 100
    arr = openeeg.read(start, start + 200)
    assert arr.shape[-1] == 100

    openeeg.close()

################
# WRITER TESTS #
################

def test_write_header(channels=[0,3]):
    """Writes a file using a subset of channels from a supplied file and
    then tests header equality."""

    openeeg = openEDF(DATAPATH)
    header = openeeg.header
    
    with openWriter(WRITEPATH) as outfile:
        outfile.write(header, openeeg, channels=channels)
    
    # reopen and test file
    with openEDF(WRITEPATH) as infile:
        h = header.filter(channels)
        assert infile.header == h

    openeeg.close()

def test_write_data():
    """Test if sequentially written data for a selection of channels matches
    the supplied data."""
    
    openeeg = openEDF(DATAPATH)
    openeeg2 = openEDF(WRITEPATH)

    #read data in steps of 5 million samples
    starts = np.arange(0, openeeg.shape[-1], int(5e6))
    stops = starts + int(5e6)
    for start, stop in zip(starts, stops):
        arr = openeeg.read(start, stop, channels=[0,3])
        other = openeeg2.read(start, stop)
        assert np.allclose(arr, other)









