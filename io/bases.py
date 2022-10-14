"""A collection of base classes for reading & writing EEG data & annotations

The abstract base classes of this module define interfaces for reading and
writing EEG data and associated annotations. Inheritors of these classes 
must supply all abstract methods to create a fully instantiable object.

Typical usage example:

These abstract classes are not part of the public interface and can not
be instantiated.
"""

import abc
import pprint
from inspect import getmembers
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from openseize.core import mixins


class Header(dict):
    """Base class for reading & storing an EEG file's header data.

    This base class defines the expected interface for all format specific
    headers. It is not instantiable. Inheriting classes must override the
    bytemap method.
    """

    def __init__(self, path):
        """Initialize this Header.

        Args:
            path: Path instance or str
                A path to eeg data file.
        """

        self.path = Path(path) if path else None
        dict.__init__(self)
        self.update(self.read())

    def __getattr__(self, name):
        """Provides '.' notation access to this Header's values.
        
        Args:
            name: str
                String name of a header bytemap key or header property.
        """

        try:
            return self[name]
        except:
            msg = "'{}' object has not attribute '{}'"
            raise AttributeError(msg.format(type(self).__name__, name))

    def bytemap(self):
        """Returns a format specific mapping specifying byte locations 
        in an eeg file containing header data."""

        raise NotImplementedError

    def read(self, encoding='ascii'):
        """Reads the header of a file into a dict using a file format
        specific bytemap.

        Args:
            encoding: str         
                The encoding used to write the header data. Default is ascii
                encoding.
        """

        header = dict()
        if not self.path:
            return header

        with open(self.path, 'rb') as fp:
            # loop over the bytemap, read and store the decoded values
            for name, (nbytes, dtype) in self.bytemap().items():
                res = [dtype(fp.read(n).strip().decode(encoding=encoding))
                       for n in nbytes]
                header[name] = res[0] if len(res) == 1 else res
        return header

    @classmethod
    def from_dict(cls, dic):
        """Alternative constructor that creates a Header instance from
        a dictionary.

        Args:
            dic: dictionary 
                A dictionary containing all expected bytemap keys.
        """

        raise NotImplementedError

    def _isprop(self, attr):
        """Returns True if attr is a property of this Header.

        Args:
            attr: str
                The name of a Header attribute.
        """

        return isinstance(attr, property)

    def __str__(self):
        """Overrides dict's print string to show accessible properties."""

        # get header properties and return  pprinter fmt str
        props = [k for k, v in getmembers(self.__class__, self._isprop)]
        pp = pprint.PrettyPrinter(sort_dicts=False, compact=True)
        props = {'Accessible Properties': props}
        return pp.pformat(self) + '\n\n' + pp.pformat(props)


class Reader(abc.ABC, mixins.ViewInstance):
    """Base class for reading EEG data.

    This ABC defines all EEG data readers as context managers that read data
    from a file path. Inheritors must override the 'read' abstract method.
    """

    def __init__(self, path, mode):
        """Initialize this reader.

        Args:
            path: Path instance or str
                A path to eeg file.
            mode: str
                A mode for reading the eeg file. Must be 'r' for plain
                string files and 'rb' for binary files.
        """

        self.path = Path(path)
        self._fobj = open(path, mode)

    @abc.abstractproperty
    def channels(self):
        """Returns the channels that this Reader will read."""

    @channels.setter
    @abc.abstractmethod
    def channels(self, val):
        """Sets the channels that this Reader will read."""

    @abc.abstractmethod
    def read(self, start, stop, **kwargs):
        """Returns a numpy array of sample values between start and stop for
        each channel in channels.

        Args:
            start: int
                Start sample index of file read.
            stop: int
                Stop sample index of file read (exclusive).
            
        Returns: 
            A channels x (stop-start) array of sample values.
        """

    def __enter__(self):
        """Return reader instance as target variable of this context."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """On context exit, close this reader's file object and propogate
        errors by returning None."""

        self.close()

    def close(self):
        """Close this reader instance's opened file object."""

        self._fobj.close()


class Writer(abc.ABC, mixins.ViewInstance):
    """Base class for all writers of EEG data.

    This ABC defines all EEG writers as context managers that use a 'write'
    method to write data to a path."""

    def __init__(self, path, mode):
        """Initialize this Writer.

        Args:
            path: str or Path instance
                A path where edf file will be written to.
            mode: str
                A mode string describing if data is text format ('w') or
                binary ('wb') format. 
        """

        self.path = Path(path)
        self._fobj = open(path, mode)

    @abc.abstractmethod
    def write(self, header, data, channels, **kwargs):
        """Writes a header and numerical data for each channel in channels
        to this Writer's opened file instance.

        Args:
            header: dict
                A mapping of file specific metadata. See io.headers for
                further details.
            data: array or reader like
                A 2-D array instance, memmap file or Reader like iterable
                that returns arrays of shape channels X samples.
            channels: sequence
                A sequence of channel indices that will be written to this
                Writer's save path.
        """

    def __enter__(self):
        """Return instance as target variable of this context."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close this instances file object & propagate any error by 
        returning None."""

        self.close()

    def close(self):
        """Close this instance's opened file object."""

        self._fobj.close()


@dataclass
class Annotation:
    """An object for storing a predefined set of annotation attributes that
    can be updated with user defined attributes after object creation.
    
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


class Annotations(abc.ABC):
    """Base class for reading annotation data.

    Annotation data may be stored in a variety of formats; csv files,
    pickled objects, etc. This ABC defines the interface expected of all
    annotation readers. Specifically, all Annotations are context managers
    and all inheritors must supply abstract methods.
    """

    def __init__(self, path, **kwargs):
        """Initialize this Annotations reader.

        Args:
            path: str or Path instance
                A path location to an annotation file.
            **kwargs: dict
                Keyword args provided to Annotations open method for opening
                file at path
        """

        self._fobj, self._reader = self.open(path, **kwargs)
    
    @abc.abstractmethod
    def open(self, path, **kwargs):
        """Opens a file at path returning a file handle & row iterator."""

    @abc.abstractmethod
    def label(self, row):
        """Reads the annotation label at row in this file."""
        
    @abc.abstractmethod
    def time(self, row):
        """Reads annotation time in secs from recording start at row."""

    @abc.abstractmethod
    def duration(self, row):
        """Returns the duration of the annotated event in secs."""

    @abc.abstractmethod
    def channel(self, row):
        """Returns the sensor this annotated event was detected on."""

    def read(self, labels=None):
        """Reads annotations with labels to a list of Annotation instances.

        Args:
            labels: sequence
                A sequence of annotation string labels for which Annotation
                instances will be returned. If None, return all.

        Returns:
            A list of Annotation dataclass instances (see Annotation).
        """
        
        result = []
        names = ['label', 'time', 'duration', 'channel']
        for row in self._reader:
            ann = Annotation(*[getattr(self, name)(row) for name in names])
            if labels is None:
                result.append(ann)
            else:
                result.append(ann) if ann.label in labels else None
        return result

    def __enter__(self):
        """Return this instance as target variable of this context."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Closes this instance's file obj. & propagate errors by returning
        None."""

        self.close()

    def close(self):
        """Closes this instance's opened file object."""

        self._fobj.close()

