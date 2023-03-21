"""A collection of base classes for reading & writing EEG data & annotations

The abstract base classes of this module define interfaces for reading and
writing EEG data and associated annotations. Inheritors of these classes
must supply all abstract methods to create a fully instantiable object.

Typical usage example:

These abstract classes are not part of the public interface and can not
be instantiated.
"""

import abc
from dataclasses import dataclass
from inspect import getmembers
from pathlib import Path
import pprint
import typing

import numpy as np
import numpy.typing as npt

from openseize.core import mixins


class Header(dict):
    """An extended dictionary base class for reading EEG headers.

    This base class defines the expected interface for all readers of EEG
    headers.  Inheriting classes are required to override the bytemap method
    to be instantiable. The bytemap method is required to return a dict that
    specifies the header field string, the number of integer bytes encoding
    the value and the datatype of the value. An example bytemap dict
    looks like:
    {field_name1: (bytes_to_read, dtype), fieldname2: ...}

    Attributes:
        path:
            A python path instance to an EEG datafile with header.
    """

    def __init__(self,
                 path: typing.Optional[typing.Union[str, Path]]
    ) -> None:
        """Initialize this Header.

        Args:
            path:
                A string or python Path instance to an EEG file.
        """

        self.path = Path(path) if path else None
        dict.__init__(self)
        self.update(self.read())

    def __getattr__(self, name: str):
        """Provides '.' notation access to this Header's values.

        Args:
            name:
                Name of field to access in this Header.
        """

        try:
            return self[name]
        except Exception as exc:
            msg = "'{}' object has not attribute '{}'"
            msg.format(type(self).__name__, name)
            raise AttributeError(msg) from exc

    def bytemap(self):
        """Returns a format specific mapping specifying byte locations
        in an eeg file containing header data."""

        raise NotImplementedError

    def read(self, encoding: str = 'ascii') -> dict:
        """Reads the header of a file into a dict using a file format
        specific bytemap.

        Args:
            encoding:
                The encoding used to represent the values of each field.
        """

        header: typing.Dict[str, typing.Tuple] = {}
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
    def from_dict(cls, dic: typing.Dict) -> 'Header':
        """Alternative constructor that creates Header instance from a
        bytemap dictionary.

        Args:
            dic:
                A bytemap dictionary to construct a Header instance from.
        """

        raise NotImplementedError

    def _isprop(self, attr: str) -> bool:
        """Returns True if attr is a property of this Header.

        Args:
            attr:
                The name of a attribute field.
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
    """Abstract base class for reading EEG data.

    This ABC defines a protocol for reading EEG data from any file type.
    Specifically, all EEG readers support opening EEG files under context
    management or as an open file whose resources should be closed when
    finished. Inheritors must override the 'read' abstract method.

    Attributes:
        path:
            Python path instance to EEG file.
        mode:
            String file mode option for 'open' builtin. Must be 'r' for
            plain text files or 'rb' for binary file types.
        kwargs:
            Additional kwargs needed for opening the file at path.
    """

    def __init__(self, path: typing.Union[str, Path], mode: str, **kwargs: str
    ) -> None:
        """Initialize this reader.

        Args:
            path:
                Python path instance to an EEG data file.
            mode:
                A mode for reading the eeg file. Must be 'r' for plain
                text files and 'rb' for binary files.
            kwargs:
                Any additional kwargs are routed to the open method.
        """

        self.path = Path(path)
        self.mode = mode
        self.kwargs = kwargs
        self._fobj = None
        self.open()

    def open(self):
        """Opens the file at path for reading & stores the file descriptor to
        this Reader's '_fobj' attribute."""

        # allow Readers to read with or without context management
        # pylint: disable-next=consider-using-with, unspecified-encoding
        self._fobj = open(self.path, self.mode, **self.kwargs)

    @property
    @abc.abstractmethod
    def channels(self):
        """Returns the channels that this Reader will read."""

    @channels.setter
    @abc.abstractmethod
    def channels(self, val: int):
        """Sets the channels that this Reader will read."""

    @property
    @abc.abstractmethod
    def shape(self):
        """Returns the summed shape of all arrays the Reader will read."""

    @abc.abstractmethod
    def read(self, start: int, stop:int) -> npt.NDArray[np.float64]:
        """Returns a numpy array of sample values between start and stop for
        each channel in channels.

        Args:
            start:
                Start sample index of file read.
            stop:
                Stop sample index of file read (exclusive).

        Returns:
            A channels x (stop-start) array of sample values.
        """

    def __enter__(self):
        """Return reader instance as target variable of this context."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """On context exit, close this reader's file object and propagate
        errors by returning None."""

        self.close()

    def close(self):
        """Close this reader instance's opened file object and destroy the
        reference to the file object.

        File descriptors whether opened or closed are not serializable. To
        support concurrent processing we close & remove all references to the
        file descriptor on close.
        """

        if self._fobj:
            self._fobj.close()
            self._fobj = None


class Writer(abc.ABC, mixins.ViewInstance):
    """Abstract base class for all writers of EEG data.

    This ABC defines all EEG writers as context managers that write data to
    a file path. Inheritors must override the 'write' method.

    Attributes:
        path:
            Python path instance to an EEG data file.
        mode:
            A str mode for writing the eeg file. Must be 'r' for plain
            text files and 'rb' for binary files.
    """

    def __init__(self, path: typing.Union[str, Path], mode: str) -> None:
        """Initialize this Writer.

        Args:
            path:
                A Python path instance of where data will be written to.
            mode:
                String mode indicating if written file type should be
                plain text 'w' or binary 'wb'.
        """

        self.path = Path(path)
        self.mode = mode
        self._fobj = None

    def __enter__(self):
        """Return instance as target variable of this context."""

        # the file may be raw bytes so leave encoding unspecified
        # pylint: disable-next=unspecified-encoding
        self._fobj = open(self.path, self.mode, encoding=None)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close this instances file object & propagate any error by
        returning None."""

        if self._fobj:
            self._fobj.close()

    @abc.abstractmethod
    def write(self, *args, **kwargs) -> None:
        """Writes metadata & data to this Writer's opened file instance."""


@dataclass
class Annotation:
    """An object for storing a predefined set of annotation attributes that
    can be updated with user defined attributes after object creation.

    Attributes:
        label (str):
            The string name of this annotation.
        time (float):
            The time this annotation was made in seconds relative to the
            recording start.
        duration (float):
            The duration of this annotation in seconds.
        channel (Any):
            The string name or integer index of the channel this annotation
            created from.
    """

    label: str
    time: float
    duration: float
    channel: typing.Any


class Annotations(abc.ABC):
    """Abstract base class for reading annotation data.

    Annotation data may be stored in a variety of formats; csv files,
    pickled objects, etc. This ABC defines all annotation readers as context
    managers that read annotation files. Inheritors must override: open,
    label, time, duration and channel methods.

    Attributes:
        path:
            Python path instance to an annotation file.
        **kwargs:
            Any valid kwarg for concrete 'open' method.
    """

    def __init__(self, path: typing.Union[str, Path], **kwargs) -> None:
        """Initialize this Annotations reader.

        Args:
            path:
                A path location to an annotation file.
            **kwargs:
                Any valid kwarg for a subclasses 'open' method.
        """

        self.path = path
        self.kwargs = kwargs

    def __enter__(self):
        """Return this instance as target variable of this context."""

        # pylint: disable-next=attribute-defined-outside-init
        self._fobj, self._reader = self.open(self.path, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Closes this instance's file obj. & propagate errors by returning
        None."""

        if self._fobj:
            self._fobj.close()

    @abc.abstractmethod
    def open(self, path: Path) -> typing.Tuple[typing.IO, typing.Iterable]:
        """Opens a file at path returning a file handle & row iterator."""

    @abc.abstractmethod
    def label(self, row: typing.Iterable) -> str:
        """Reads the annotation label at row in this file."""

    @abc.abstractmethod
    def time(self, row: typing.Iterable) -> float:
        """Reads annotation time in secs from recording start at row."""

    @abc.abstractmethod
    def duration(self, row: typing.Iterable) -> float:
        """Returns the duration of the annotated event in secs."""

    @abc.abstractmethod
    def channel(self, row: typing.Iterable) -> typing.Union[int, str]:
        """Returns the channel this annotated event was detected on."""

    def read(self,
             labels: typing.Optional[typing.Sequence[str]] = None,
    ) -> typing.List[Annotation]:
        """Reads annotations with labels to a list of Annotation instances.

        Args:
            labels:
                A sequence of annotation string labels for which Annotation
                instances will be returned. If None, return all.

        Returns:
            A list of Annotation dataclass instances (see Annotation).
        """

        labels = [labels] if isinstance(labels, str) else labels

        result = []
        names = ['label', 'time', 'duration', 'channel']
        for row in self._reader:
            ann = Annotation(*[getattr(self, name)(row) for name in names])

            if labels is None:
                result.append(ann)

            elif ann.label in labels:
                result.append(ann)

            else:
                continue

        return result
