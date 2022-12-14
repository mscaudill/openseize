from pathlib import Path

import pytest

from openseize.demos import paths
from openseize.file_io import edf

@pytest.fixture(scope='session')
def demo_data():
    """ """

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



