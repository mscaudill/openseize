import inspect
from pathlib import Path

def demopaths():
    """Returns all the paths in the data dir of the demo module of
    openseize."""

    #get path to the demo data dir
    current = Path(inspect.getabsfile(inspect.currentframe()))
    data_dir = current.parent.joinpath('data')
    return {path.name: path for path in data_dir.iterdir()}


# make demo data paths available in demos namespace
paths = demopaths()
