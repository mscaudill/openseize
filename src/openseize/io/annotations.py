"""Tools for reading EEG annotations files and creating boolean arrays for
masking EEG data producers.

This module contains the following classes and functions:

    Pinnacle:
        A Pinnacle Technoologies© annotation file reader.

        Typical usage example:
        with Pinnacle(<*.txt>, start=6) as infile:
            annotations = infile.read(labels=['rest', 'exploring'])

        returns a sequence of Annotation instances from file which contain
        either the 'rest' or 'exploring' label.

    as_mask:
        A function that converts a sequence of annotations into a boolean
        array for masking a producer of data values.
"""

import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from openseize.io.bases import Annotations
from openseize.core import arraytools


class Pinnacle(Annotations):
    """A Pinnacle Technologies© annotations file reader.

    Pinnacle files store annotation data to a plain text file. This
    reader reads each row of this file extracting and storing annotation
    data to a sequence Annotation objects one per annotation (row) in the
    file.
    """

    def open(self, path, start=0, delimiter='\t', **kwargs):
        """Opens a file returning a file handle and row iterator.
    
        Args:
            path: str or Path instance
                A annotation file path location.
            start: int
                The row number of the column headers in the file.
            **kwargs: A valid keyword argument for CSV.DictReader builtin.
        """

        fobj = open(Path(path))
        #advance to start row and return a reader
        [next(fobj) for _ in range(start)]
        return fobj, csv.DictReader(fobj, delimiter=delimiter, **kwargs)

    def label(self, row):
        """Extracts the annotation label for a row in this file."""

        return row['Annotation']

    def time(self, row):
        """Extracts the annotation time of a row of this file."""

        return float(row['Time From Start'])

    def duration(self, row):
        """Measures the duration of an annotation for a row in this file."""

        #compute time difference from start and stop datetime objs
        fmt = '%m/%d/%y %H:%M:%S.%f'
        start = datetime.strptime(row['Start Time'], fmt)
        stop = datetime.strptime(row['End Time'], fmt)
        return (stop - start).total_seconds()

    def channel(self, row):
        """Extracts the annotation channel for a row in this file."""

        return row['Channel']


def as_mask(annotations, size, fs, include=True):
    """Convert a sequence of annotation objects into a 1-D boolean array. 

    Args:
        annotations: list
            A sequence of annotation objects to convert to a mask.
        size: int
            The length of the boolean array to return.
        fs: int
            The sampling rate in Hz of the recorded EEG.
        include: bool
            A boolean to determine if annotations should be set to True or
            False in the returned array. Default is True, meaning all values
            are False in the returned array except for samples where the
            annotations are located.

    Returns:
        A 1-D boolean array of length size.
    """

    epochs = [(ann.time, ann.time + ann.duration) for ann in annotations]
    samples = np.round(np.array(epochs) * fs).astype(int)
    slices = [slice(*pts) for pts in samples]
    result = arraytools.filter1D(size, slices)
    return result if include else ~result

