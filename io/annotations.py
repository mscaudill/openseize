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
# FIXME This file contains some specific readers that should be moved to
# a project specific directory for SWD detection/classification at a later
# date. Each of these readers is marked with a FIXME.


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
            are False in the returned array expect for samples where the
            annotations are located.

    Returns:
        A 1-D boolean array of length size.
    """

    epochs = [(ann.time, ann.time + ann.duration) for ann in annotations]
    samples = np.round(np.array(epochs) * fs).astype(int)
    slices = [slice(*pts) for pts in samples]
    result = arraytools.filter1D(size, slices)
    return result if include else ~result


# FIXME Move to project specific package later
import scipy.io as sio
class XueMat(Annotations):
    """A Xue lab MAT file annotations reader.

    The MAT file storing annotated events in the Xue lab contains the start
    and stop times. No channel or label data is present so we will set the
    labels to be 'SWD' and the channel to 'ALL'
    """

    def open(self, path, name='DEL_ts'):
        """Opens a MAT file containing an array of times.

        Args:
            path: str or Path instance
                A annotation file path location.
            name: str
                Name of the stored array of times in the MAT file. Defaults
                to 'DEL_ts'
        """

        fobj = open(Path(path))
        return fobj, sio.loadmat(path)[name]

    def label(self, row):
        """Extracts the annotation label for a row in this array."""

        return 'SWD'

    def time(self, row):
        """Extracts the annotation time of a row of this array."""

        return np.around(row[0], decimals=3)

    def duration(self, row):
        """Return the annotation duration in secs for this row of array."""

        return np.around(row[1] - row[0], decimals=3)

    def channel(self, row):
        """Extracts the annotation channel for a row in this file."""

        return 'ALL'

# FIXME Move to project specific package later
class Frankel(Annotations):
    """A Frankel lab CSV Annotations file reader."""

    def open(self, path, start=0, delimiter=',', **kwargs):
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

    def rate(self, row):
        """Computes the sample rate of recording for a Frankel EEG.

        The Frankel lab store annotations at samples instead of secs so we
        use the recording length and duration in seconds to compute the
        sample rate to convert samples to time.
        """

        return float(row['Sample Length']) /  float(row['Duration (s)'])

    def label(self, row):
        """Return the label of this rows annotation."""

        return row['Analyzed Event']

    def time(self, row):
        """Return the annotation time in secs from this row of reader."""

        return float(row['Start Sample']) / self.rate(row)

    def duration(self, row):
        """Returns annotation duration in secs from this row of reader."""

        return float(row['Duration (s)'])

    def channel(self, row):
        """Returns channel of this annotation from this row of reader."""

        return row['Channel Index']


# FIXME REMOVE TO TESTING CODE
if __name__ == '__main__':

    fp = ('/home/matt/python/nri/data/rett_eeg/dbs_treated/annotations/'
          '5872_Left_group A-D.txt')
    with Pinnacle(fp, start=6, delimiter='\t') as p:
        annotations0 = p.read(labels='exploring')


    fp2 = '/home/matt/python/nri/data/openseize/CW0101_P227.mat'
    with XueMat(fp2) as f:
        annotations1 = f.read()

    fp3 = ('/home/matt/python/nri/data/frankel_lab/annotations/'
           'Scn8a_Gria4x2_Gabrg2_M1_14242.csv')
    with Frankel(fp3, start=2) as fp:
        annotations2 = fp.read()



