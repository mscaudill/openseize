from pathlib import Path

import pytest
import numpy as np

from openseize.demos import paths
from openseize.file_io import edf

@pytest.fixture(scope='session')
def demo_data():
    """Returns a filepath to the demo data downloading the data if
    needed."""

    return paths.locate('recording_001.edf', dialog=False)

@pytest.fixture(scope='session')
def written_data(demo_data):
    """ """

    fp = paths.data_dir.joinpath('write_test.edf')
    if fp.exists():
        return fp

    channels = [0,3]
    with edf.Reader(demo_data) as reader:
        with edf.Writer(fp) as writer:
            writer.write(reader.header, reader, channels=channels)
    return fp

@pytest.fixture(scope='session')
def irregular_written_data(demo_data):
    """ """

    # The edf will contain 200 data records with varying samples per record
    # for the different channels
    seed = 0
    nrecords = 200
    samples_per_record = [5000, 10000, 20000, 5000]
    pad_value = np.nan

    # create random data array to write
    rng = np.random.default_rng(seed)
    num_chs = len(samples_per_record)
    x = pad_value * np.ones((num_chs, max(samples_per_record) * nrecords))
    for ch in range(num_chs):
        stop = samples_per_record[ch] * nrecords
        x[ch, :stop] = 1000 * rng.random(samples_per_record[ch] * nrecords)

    fp = paths.data_dir.joinpath('irregular_write_test.edf')
    if fp.exists():
        return fp, x

    # write the data using demo_data header
    with edf.Reader(path=demo_data) as reader:
        header = reader.header.filter(list(range(num_chs)))
        header['num_records'] = nrecords
        header['samples_per_record'] = samples_per_record
    
    with edf.Writer(fp) as writer:
        writer.write(header, x, channels=list(range(num_chs)))

    return fp, x

@pytest.fixture(scope='session')
def split_data(irregular_written_data):
    """ """

    unsplit_fp, _ = irregular_written_data

    mapping={'split0':[0,1], 'split1':[2,3]}
    edf.splitter(unsplit_fp, mapping)

    # create a dict of split filepaths and channels in each split
    result = {}
    for fname, indices in mapping.items():
        fp = unsplit_fp.parent.joinpath(fname).with_suffix('.edf')
        result[fp] = indices
    return result



