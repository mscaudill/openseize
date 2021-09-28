import csv
import abc
from datetime import datetime
from pathlib import Path
import numpy as np
import scipy.io as sio

from openseize.io.managers import FileManager
from openseize.mixins import ViewContainer

class Annotation(ViewContainer):
    """An immutable obj for holding annotation data.

    Attrs:
        label (str):        str name for this annotation (e.g 'artifact')
        start_time (

    This object uses slots to reduce memory usage for possibly many
    annotations to be stored.
    """

    __slots__ = ['label', 'start_time', 'duration', 'channel']

    def __init__(self, label, start_time, duration, channel):
        """Initialize this annotation."""

        self.label = label
        self.start_time = start_time
        self.duration = duration
        self.channel = channel


class Annotations(FileManager, abc.ABC):
    """ABC defining expected interface of all annotation file managers.

    All annotation file managers must provide methods for computing the
    attributes of a single annotation instance (see Annotation) from a row
    read by the csv DictReader builtin. 
    """

    def __init__(self, path, start_row=0, **kwargs):
        """Initialize this annotation FileManager by opening the file,
        building a csv DictReader and advancing to start row of file."""

        super().__init__(path, 'r')
        [next(self._fobj) for _ in range(start_row)]
        self._reader = csv.DictReader(self._fobj, **kwargs)

    @abc.abstractmethod
    def label(self, row):
        """Returns an annotation name for a row in a csv file."""
        
    @abc.abstractmethod
    def event_time(self, row):
        """Returns annotation time in secs relative to recording start."""

    @abc.abstractmethod
    def duration(self, row):
        """Returns the duration of the annotated event in secs."""

    @abc.abstractmethod
    def channel(self, row):
        """Retrurns the sensor this annotated event was detected on."""
    
    def read(self, labels=None):
        """Returns a list of Annotation objects for this CSV.

        Args:
            labels (list):      seq of annotation labels to constrain 
                                which annotations will be returned. If None,
                                return all annotated events (Default = None)
        """

        result = []
        for row in self._reader:
            attrs = [getattr(self, attr)(row) for attr in 
                     ['label','event_time', 'duration', 'channel']]
            result.append(Annotation(*attrs))
        if labels:
            result = [ann for ann in result if ann.label in labels]
        return result


class Pinnacle(Annotations):
    """A FileManager for reading Pinnacle formatted annotations."""

    def label(self, row):
        """Returns the annotation label of this row in the file."""
       
        return row['Annotation']

    def event_time(self, row):
        """Returns annotation time in secs of this row in the file."""

        return float(row['Time From Start'])

    def duration(self, row):
        """Returns annotation duration in secs of this row in the file."""

        #create a format for datetime objs
        fmt = '%m/%d/%y %H:%M:%S.%f'
        start = datetime.strptime(row['Start Time'], fmt)
        stop = datetime.strptime(row['End Time'], fmt)
        return (stop - start).total_seconds()

    def channel(self, row):
        """Returns the annotation channels of this row in the file."""

        return row['Channel']


class XueMat(Annotations):
    """A FileManager for reading MAT annotation files in Xue lab format

    The XueMat format only stores the start and stop times in secs, no 
    labels or channels. However, we require all annotation readers to be
    FileManagers that fullfill all the required abstract methods of 
    Annotations. This creates a consistent interface for annotation reading.
    """

    def __init__(self, path, name='DEL_ts'):
        """ """

        #For consistency open the file, this handle will go unused
        self._fobj = open(Path(path), 'r')
        #reader is just an array in Xue fmt
        self._reader = sio.loadmat(path)[name]

    def label(self, row):
        """All annotations in the XueMat fmt are just TrueEvents."""

        return 'TrueEvent'

    def event_time(self, row):
        """Return the annotation time in secs from this row of reader."""

        return np.around(row[0], decimals=3)

    def duration(self, row):
        """Return the annotation duration in secs for this row of reader."""

        return np.around(row[1] - row[0], decimals=3)

    def channel(self, row):
        """Returns channel of this annotation from this row of reader."""

        return 'ANY'


class Frankel(Annotations):
    """A FileManager for reading Frankel lab annotations."""

    def rate(self, row):
        """ """

        return float(row['Sample Length']) /  float(row['Duration (s)'])

    def label(self, row):
        """Return the label of this rows annotation."""

        return row['Analyzed Event']

    def event_time(self, row):
        """Return the annotation time in secs from this row of reader."""

        return float(row['Start Sample']) / self.rate(row)

    def duration(self, row):
        """Returns annotation duration in secs from this row of reader."""

        return float(row['Duration (s)'])

    def channel(self, row):
        """Returns channel of this annotation from this row of reader."""

        return row['Channel Index']

if __name__ == '__main__':
    


    #build path and try to open Qis data 
    fp = ('/home/matt/python/nri/data/rett_eeg/dbs_treated/annotations/'
          '5872_Left_group A-D.txt')
    fp2 = '/home/matt/python/nri/data/openseize/CW0101_P227.mat'

    fp3 = ('/home/matt/python/nri/data/frankel_lab/annotations/'
           'Scn8a_Gria4x2_Gabrg2_M1_14242.csv')

    """
    with Pinnacle(fp, start_row=6, delimiter='\t') as p:
        annotations = p.read(labels='exploring')
    """

    """
    with XueMat(fp2) as f:
        annotations = f.read()
    """

    with Frankel(fp3, start_row=2) as fp:
        annotations = fp.read()


