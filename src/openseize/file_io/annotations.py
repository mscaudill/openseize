"""A collection of annotation readers and annotation masking tools.

Annotations are useful for removing artifacts, filtering data by user
labeled states and more. This module provides readers for reading
annotations. Additionally, these annotations can be converted into boolean
mask to provide to Openseize's producers for masking produced data.

This module contains the following classes and functions:
```
    Pinnacle:
        A Pinnacle Technoologies© annotation file reader.

    as_mask:
        A function that creates boolean mask from sequences of annotations
        that can be used to filter the data produced from a producer
        instance.
```

Examples:
        >>> # read the annotations from the demo annotation file
        >>> from openseize.demos import paths
        >>> filepath = paths.locate('annotations_001.txt')
        >>> from openseize.io.annotations import Pinnacle
        >>> # read all the annotations
        >>> with Pinnacle(filepath, start=6) as pinnacle:
        >>>     annotations = pinnacle.read()
        >>> print(annotations)
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import cast, Dict, IO, Iterable, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from openseize.core import arraytools
from openseize.file_io.bases import Annotation
from openseize.file_io.bases import Annotations


class Pinnacle(Annotations):
    """A reader of Pinnacle Technologies© annotation csv files.

    This annotation reader's 'read' method reads annotations into a list of
    Annotation dataclass instances. Each Annotation dataclass has the
    following attributes:

    - label: A string label given to an annotation.
    - time: Time, relative to recording start, in secs of an annotation.
    - duration: The duration in seconds of an annotation.
    - channel: The channel(s) an annotation was detected on.

    Attributes:
        path:
            Python path instance to Pinnacle© file.
        kwargs:
            Any valid kwarg for csv.DictReader initializer.

    Examples:
        >>> # read the annotations from the demo annotation file
        >>> from openseize.demos import paths
        >>> filepath = paths.locate('annotations_001.txt')
        >>> from openseize.io.annotations import Pinnacle
        >>> # read the 'rest' and 'exploring' annotations
        >>> with Pinnacle(filepath, start=6) as pinnacle:
        >>>     annotations = pinnacle.read(labels=['rest', 'exploring'])
        >>> # get the first annotation and print it
        >>> print(annotations[0])
        >>> # print the first annotations duration
        >>> print(annotations[0].duration)
    """

    def open(self,
            path: Union[str, Path],
            start: int = 0,
            delimiter: str ='\t',
            **kwargs
    ) -> Tuple[IO[str], Iterable[dict]]:
        """Opens a Pinnacle formatted CSV annotation file.

        Called by 'Annotations.__init__' to initialize this Pinnacle
        context manager.

        Args:
            path:
                A annotation file path location.
            start:
                The row number of the column headers in the file.
            delimiter:
                The string character separating columns in the file.
            **kwargs:
                Any valid keyword argument for CSV.DictReader builtin.

        Returns:
            A tuple (file_obj, DictReader) where file_obj is the open file
            instance and DictReader is the builtin csv DictReader.
        """

        # This method is called within context management (see base class)
        # pylint: disable-next=consider-using-with
        fobj = open(Path(path), encoding='utf-8')
        # advance to start row and return a reader
        # pylint: disable-next=expression-not-assigned
        [next(fobj) for _ in range(start)]
        return fobj, csv.DictReader(fobj, delimiter=delimiter, **kwargs)

    def label(self, row:  Dict[str, str]) -> str:
        """Extracts the annotation label for a row in this file."""

        return row['Annotation']

    def time(self, row: Dict[str, str]) -> float:
        """Extracts the annotation time of a row of this file."""

        return float(row['Time From Start'])

    def duration(self, row: Dict[str, str]) -> float:
        """Measures the duration of an annotation for a row in this file."""

        #compute time difference from start and stop datetime objs
        fmt = '%m/%d/%y %H:%M:%S.%f'
        start = datetime.strptime(row['Start Time'], fmt)
        stop = datetime.strptime(row['End Time'], fmt)
        return (stop - start).total_seconds()

    def channel(self, row: Dict[str, str]) -> Union[int, str]:
        """Extracts the annotation channel for a row in this file."""

        return row['Channel']


def as_mask(annotations: Sequence[Annotation],
            size: int,
            fs: float,
            include: bool = True,
) -> npt.NDArray[np.bool_]:
    """Creates a boolean mask from a sequence of annotation dataclass
    instances..

    Producers of EEG data may receive an optional boolean array mask.  This
    function creates a boolean mask from a sequence of annotations and is
    therefore useful for filtering EEG data by annotation label during
    processing.

    Args:
        annotations:
            A sequence of annotation dataclass instances to convert to a 
            mask.
        size:
            The length of the boolean array to return.
        fs:
            The sampling rate in Hz of the digital system.
        include:
            Boolean determining if annotations should be set to True or
            False in the returned array. True means all values
            are False in the returned array except for samples where the
            annotations are located.

    Returns:
        A 1-D boolean array of length size.

    Examples:
        >>> # read the annotations from the demo annotation file
        >>> from openseize.demos import paths
        >>> filepath = paths.locate('annotations_001.txt')
        >>> from openseize.io.annotations import Pinnacle
        >>> # read the 'rest' annotations
        >>> with Pinnacle(filepath, start=6) as pinnacle:
        >>>     annotations = pinnacle.read(labels=['rest'])
        >>> # create a mask measuring 3700 secs at 5000 Hz
        >>> mask = as_mask(annotations, size=3700*5000, fs=5000)
        >>> # measure the total time in secs of 'rest' annotation
        >>> print(np.count_nonzero(mask) / 5000)
        15.0
    """

    epochs = [(ann.time, ann.time + ann.duration) for ann in annotations]
    samples = np.round(np.array(epochs) * fs).astype(int)
    slices = [slice(*pts) for pts in samples]
    result = arraytools.filter1D(size, slices)
    result = result if include else ~result
    return cast(np.ndarray, result)
