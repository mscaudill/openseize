"""

"""

import abc
from dataclasses import dataclass
from typing import Any
from pathlib import Path

@dataclass(frozen=True)
class Annotation:
    """An immutable object for storing annotation data.

    Attributes:
        label: str
            The string name of this annotation.
        time: float
            The time this annotation was made in seconds relative to the
            recording start.
        duration: float
            The duration of this annotation in seconds.
        channel: Any
            The string name or integer index of the channel this annotation
            created from.
    """
    
    label: str
    time: float
    duration: float
    channel: Any


def to_bool(annotations, size, fs, include=True):
    """ """

    pass


class AnnotationReader(abc.ABC):
    """ """

    # FIXME Move to BASES
    def __init__(self, path, mode='r', **kwargs):
        """Initialize this Reader."""

        self.path = Path(path)
        self.open(mode, **kwargs)
        self._fobj, self._reader = self.open()

    @abc.abstractmethod
    def open(self, mode, **kwargs):
        """Returns an iterable over rows  """

    def __enter__(self):
        """Return this instance as target variable of this context."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Closes this instance's file obj. & propagate errors by returning
        None."""

        self.close()

    def close(self):
        """Close this reader instance's opened file object."""

        self._fobj.close()

    @abc.abstractmethod
    def label(self, row):
        """Returns an annotation name for a row in a csv file."""
        
    @abc.abstractmethod
    def time(self, row):
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
        return Annotations(result)


class Pinnacle(AnnotationReader):
    """ """

    def open(self, mode, start_row=0, **kwargs):
        """ """

        fobj = open(path, mode)
        [next(fobj) for _ in range(start_row)]
        reader = csv.DictReader(fobj, **kwargs)
        return fobj, reader

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




class MAT(AnnotationReader):
    pass



if __name__ == '__main__':

    a = Annotation(label='artifact', time=30.2, duration=2.3, channel=0)




