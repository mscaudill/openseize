"""Tools for reading and writing EEG metadata and data in the European Data
Format (EDF/EDF+). This module contains:

    - Header: An extended dictionary representation of and EDF header
    - Reader: A reader of EDF data and metadata
    - Writer: A writer of EDF data and metadata
    - splitter: A function to split EDF files into multiple EDF files.

## Header
The Header section of an EDF file is partitioned into sequential
sections containing metadata. Each section has a specified number of
bytes used to encode an an 'ascii' string. For example the first two
sections of the header are:
```
***************************************
8 bytes | 80 bytes | ................
***************************************
```
The first 8 bytes are the EDF version string and the next 80 bytes are the
patient_id string. The full specification of the EDF header can be found
here: https://www.edfplus.info/specs/edf.html. A Header instance, is an
extended dictionary keyed on the field name from the EDF specification (i.e.
version, patient, etc) with a value that has been decoded from the file at
path like this:
```
{'version': 'EDF+', 'patient': 'mouse_1', ...}
```

## Reader
EDF files are divided into header and data records sections. Each data
record contains measured signals and annotation signals stored sequentially.
Below is a sample layout of a single data record:
```
***********************************************************
Ch0 samples | Ch1 samples | Ch2 samples | ... | Annotations
***********************************************************
```
To distinguish numerical signals from annotation signals, we refer to
numerical signals as channels. Currently, readers does not support the
reading of annotation signals stored in the data records.

For details on the EDF/+ file specification please see;
https://www.edfplus.info/specs/index.html

The reader supports reading EEG data and metadata from an EDF file with and
without context management. If opened outside of context management, you
should close this Reader's instance manually by calling the 'close' method
to recover open file resources when you finish processing a file.

Examples:
        >>> # Read samples from an EDF with Context management
        >>> from openseize.demos import paths
        >>> filepath = paths.locate('recording_001.edf')
        >>> from openseize.io.edf import Reader
        >>> # open a reader using context management and reading 120 samples
        >>> # from all 4 channels
        >>> with Reader(filepath) as infile:
        >>>     x = infile.read(start=0, stop=120)
        >>> print(x.shape)
        ... (4, 120)
        >>> # open the reader without context management
        >>> reader = Reader(filepath)
        >>> # set reader to read only channels 0 and 2 data
        >>> reader.channels = [0, 2]
        >>> # read samples 0 to 99 from channels 0 and 2
        >>> y = reader.read(start=0, stop=99)
        >>> print(y.shape)
        ... (2, 99)
        >>> # Read samples from an EDF without context management
        >>> reader = Reader(filepath)
        >>> y = reader.read(start=10, stop=61)
        >>> # view the reader's Header instance
        >>> print(reader.header)
        >>> # when done with the reader, you should close it to recover
        >>> # resources
        >>> reader.close()

## Writer
A Writer is a context manager for writing EEG data and metadata to an EDF
binary file. Unlike Readers it must be opened under the context management
protocol. Importantly, this writer does not currently support writing
annotations to an EDF file.

Examples:
    >>> from openseize.demos import paths
    >>> filepath = paths.locate('recording_001.edf')
    >>> # Create a reader that will read only channels [0, 1]
    >>> # and write out these channels to a new file
    >>> writepath = paths.data_dir.joinpath('subset_001.edf')
    >>> with Reader(filepath) as reader:
    >>>     with Writer(writepath) as writer:
    >>>         writer.write(reader.header, reader, channels=[0, 1])

## Splitter
A tool for splitting an EDF into multiple EDFs each containing different
channels of the unsplit EDF. In particular, this tool is useful for
partitioning an EDF with channels from different subjects into multiple
single subject EDFs. The original 'joined' EDF is left unmodified.
"""

import copy
from pathlib import Path
from typing import (cast, Dict, Generator, List, Optional, Sequence, Tuple,
                    Union)

import numpy as np
import numpy.typing as npt

from openseize.file_io import bases


class Header(bases.Header):
    """An extended dictionary representation of an EDF Header.

    The Header section of an EDF file is partitioned into sequential
    sections containing metadata. Each section has a specified number of
    bytes used to encode an an 'ascii' string. This Header reads and
    stores each piece of metadata to a extended dict object.

    Attributes:
        path: (Path)
            A python path instance to the EDF file.
    """

    def bytemap(self, num_signals: Optional[int] = None) -> Dict:
        """A dictionary keyed on fields from the EDF specification whose
        values are a tuple containing the number of bytes used to encode the
        field's value and the datatype of the value. For example:

        {'version': (8, str), 'patient': (80, str), ....}

        Args:
            num_signals:
                The number of signals (channels & annotation) in the file.
                If None, this value will be read from the file.

        Returns:
            A dictionary of EDF field names and tuples ([bytes], dtype)
            specifying the number of bytes to read and the data type of the
            value.
        """

        if num_signals is None:
            num_signals = self.count_signals()

        return {'version': ([8], str),
                'patient': ([80], str),
                'recording': ([80], str),
                'start_date': ([8], str),
                'start_time': ([8], str),
                'header_bytes': ([8], int),
                'reserved_0': ([44], str),
                'num_records': ([8], int),
                'record_duration': ([8], float),
                'num_signals': ([4], int),
                'names': ([16] * num_signals, str),
                'transducers': ([80] * num_signals, str),
                'physical_dim': ([8] * num_signals, str),
                'physical_min': ([8] * num_signals, float),
                'physical_max': ([8] * num_signals, float),
                'digital_min': ([8] * num_signals, float),
                'digital_max': ([8] * num_signals, float),
                'prefiltering': ([80] * num_signals, str),
                'samples_per_record': ([8] * num_signals, int),
                'reserved_1': ([32] * num_signals, str)}

    def count_signals(self) -> int:
        """Returns the signal count in the EDF's header.

        The signal count will include annotation signals if present.
        """

        if not self.path:
            # if this Header was built from a dict, path is None
            return int(self.num_signals)

        with open(self.path, 'rb') as fp:
            fp.seek(252) # edf specifies num signals at 252nd byte
            return int(fp.read(4).strip().decode())

    @classmethod
    def from_dict(cls, dic: Dict) -> 'Header':
        """Alternative constructor for creating a Header from a bytemap.

        Args:
            dic:
                A dictionary containing all expected bytemap keys.
        """

        instance = cls(path=None)
        instance.update(dic)
        # validate dic contains all bytemap keys
        if set(dic) == set(instance.bytemap(1)):
            return instance

        msg='Missing keys required to create a header of type {}.'
        raise ValueError(msg.format(cls.__name__))

    @property
    def annotated(self) -> bool:
        """Returns True if the EDF header contains annotations."""

        return 'EDF Annotations' in self.names

    @property
    def annotation(self) -> Optional[int]:
        """Returns the index of the annotation signal or None if EDF header
        does not contain annotations."""

        result = None
        if self.annotated:
            result = self.names.index('EDF Annotations')
        return result

    @property
    def channels(self) -> Sequence[int]:
        """Returns the non-annotation 'ordinary signal' indices."""

        signals = list(range(self.num_signals))
        if self.annotation:
            signals.pop(self.annotation)
        return signals

    @property
    def samples(self) -> Sequence[int]:
        """Returns the total sample count of each channels in EDF header.

        The last record of the EDF may not be completely filled with
        recorded signal values depending on the software that created it.
        """

        nrecs = self.num_records
        samples = np.array(self.samples_per_record) * nrecs
        return [samples[ch] for ch in self.channels]

    @property
    def record_map(self) -> Sequence[slice]:
        """Returns a list of slice objects holding the start, stop samples
        for each channel within a data record.

        Data records in the EDF data section following the header contain
        data organized like so

        --------------------------------------------------------------
        Ch0 samples | Ch1 samples | Ch2 samples | ... | Annotations
        --------------------------------------------------------------
        (start, stop)|(start, stop)|(start, stop)| ... |(start, stop)
        --------------------------------------------------------------

        Returns:
            A slice object from start to stop for each signal in a record.
        """

        scnts = np.insert(self.samples_per_record, 0, 0)
        cum = np.cumsum(scnts)
        return list(slice(a, b) for (a, b) in zip(cum, cum[1:]))

    @property
    def slopes(self):
        """Returns a 1-D array of channel slopes (i.e channel gains).

        Notes:
            The EDF specification asserts that the physical voltage values
            'p' are linearly mapped from the integer digital values 'd'
            according to:

                p = slope * d + offset
                slope = (pmax -pmin) / (dmax - dmin)
                offset = p - slope * d for any (p,d)
        """

        pmaxes = np.array(self.physical_max)[self.channels]
        pmins = np.array(self.physical_min)[self.channels]
        dmaxes = np.array(self.digital_max)[self.channels]
        dmins = np.array(self.digital_min)[self.channels]
        return (pmaxes - pmins) / (dmaxes - dmins)

    @property
    def offsets(self):
        """Returns a 1-D array of channel offsets (i.e. intercepts).

        Notes:
            See slopes property for offset definition.
        """

        pmins = np.array(self.physical_min)[self.channels]
        dmins = np.array(self.digital_min)[self.channels]
        return pmins - self.slopes * dmins

    def filter(self, indices: Sequence[int]) -> 'Header':
        """Returns a new Header instance that contains only the metadata for
        each signal index in indices.

        Args:
            indices:
                Signal indices to retain in this Header.

        Returns:
            A new Header instance.
        """

        header = copy.deepcopy(self)
        for key, value in header.items():
            if isinstance(value, list):
                header[key] = [value[idx] for idx in indices]

        # update header_bytes and num_signals
        bytemap = self.bytemap(len(indices))
        nbytes = sum(sum(tup[0]) for tup in bytemap.values())
        header['header_bytes'] = nbytes
        header['num_signals'] = len(indices)

        return header


class Reader(bases.Reader):
    """A reader of European Data Format (EDF/EDF+) files.

    This reader supports reading EEG data and metadata from an EDF file with
    and without context management (see Introduction). If opened outside
    of context management, you should close this Reader's instance manually
    by calling the 'close' method to recover open file resources when you
    finish processing a file.

    Attributes:
        header (dict):
            A dictionary representation of the EDFs header.
        shape (tuple):
            A (channels, samples) shape tuple.
        channels (Sequence):
            The channels to be returned from the 'read' method call.

    Examples:
        >>> from openseize.demos import paths
        >>> filepath = paths.locate('recording_001.edf')
        >>> from openseize.io.edf import Reader
        >>> # open a reader using context management and reading 120 samples
        >>> # from all 4 channels
        >>> with Reader(filepath) as infile:
        >>>     x = infile.read(start=0, stop=120)
        >>> print(x.shape)
        ... (4, 120)
    """

    def __init__(self, path: Union[str, Path]) -> None:
        """Extends the Reader ABC with a header attribute."""

        super().__init__(path, mode='rb')
        self.header = Header(path)
        self._channels = self.header.channels

    @property
    def channels(self) -> Sequence[int]:
        """Returns the channels that this Reader will read."""

        return self._channels

    @channels.setter
    def channels(self, values: Sequence[int]):
        """Sets the channels that this Reader will read.

        Args:
            values:
                Sets the channels this Reader's 'read' method will return
                data from.
        """

        if not isinstance(values, Sequence):
            msg = 'Channels must be type Sequence not {}'
            raise ValueError(msg.format(type(values)))

        self._channels = values

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns a 2-tuple containing the number of channels and
        number of samples in this EDF."""

        return len(self.channels), max(self.header.samples)

    def _decipher(self,
                  arr: np.ndarray,
                  channels: Sequence[int],
                  axis: int = -1,
    ):
        """Converts decoded data record integers to float voltages.

        Physical voltage values 'p' are linearly mapped from the
        decoded integer values 'd' according to:

            p = slope * d + offset
            slope = (pmax -pmin) / (dmax - dmin)
            offset = p - slope * d for any (p,d)

        The EDF header contains pmax, pmin, dmax and dmin for each channel.

        Args:
            arr:
                An array of integer values decoded from the EDF.
            channels:
                The channel indices that were decoded. Each channel may have
                a unique slope and offset.
            axis:
                The samples axis of arr.

        Returns:
            A float64 ndarray of voltages with the same shape as 'arr'.
        """

        slopes = np.array(self.header.slopes[channels])
        offsets = np.array(self.header.offsets[channels])
        #expand to 2-D for broadcasting
        slopes = np.expand_dims(slopes, axis=axis)
        offsets = np.expand_dims(offsets, axis=axis)
        result = arr * slopes
        result += offsets
        return cast(np.ndarray, result)

    def _find_records(self,
                      start: int,
                      stop: int,
                      channels: Sequence[int],
    ) -> Sequence[Tuple[int, int]]:
        """Returns the first and last record indices that include start to
        stop samples for each channel in channels.

        Notes:
            The number of samples for each channel will be different if the
            sample rates are unequal. Thus, this method returns a first and
            last record number for each channel.

        Args:
            start:
                The start sample used to locate the first record.
            stop:
                The stop sample (exclusive) used to locate the last record.
            channels:
                The channel indices to read.

        Returns:
            A list of (first, last) record numbers for each channel.
        """

        spr = np.array(self.header.samples_per_record)[channels]
        starts = start // spr
        stops = np.ceil(stop / spr).astype('int')
        return list(zip(starts, stops))

    def _records(self, a: int, b: int):
        """Reads samples in the ath to bth record.

        If b exceeds the number of records in the EDF, then samples up to the
        end of file are returned. If a exceeds the number of records, an
        empty array is returned.

        Args:
            a:
                The first record to read.
            b:
                The last record to be read (exclusive).

        Returns:
            A ndarray of shape (b-a) * sum(samples_per_record)
        """

        if a >= self.header.num_records:
            return np.empty((1,0))
        b = min(b, self.header.num_records)
        cnt = b - a

        self._fobj.seek(0)
        #EDF samples are 2-byte little endian integers
        bytes_per_record = sum(self.header.samples_per_record) * 2
        #get offset in bytes & num samples spanning a to b
        offset = self.header.header_bytes + a * bytes_per_record
        nsamples = cnt * sum(self.header.samples_per_record)
        #read records and reshape to num_records x sum(samples_per_record)
        recs = np.fromfile(self._fobj, '<i2', nsamples, offset=offset)
        arr = recs.reshape(cnt, sum(self.header.samples_per_record))
        return arr

    def _padstack(self,
                  arrs: Sequence[np.ndarray],
                  value: float,
                  axis: int = 0
    ):
        """Returns a 2-D array from a ragged sequence of 1-D arrays.

        Args:
            arrs:
                A ragged sequence of 1-D arrays to combine.
            value:
                Padding value used to lengthen 1-D arrays.
            axis:
                The axis along which to stack the padded 1-D arrays.

        Returns:
            A 2-D array.
        """

        longest = max(len(arr) for arr in arrs)
        pad_sizes = np.array([longest - len(arr) for arr in arrs])

        if all(pad_sizes == 0):
            return np.stack(arrs, axis=0)

        x = [np.pad(arr.astype(float), (0, pad), constant_values=value)
                for arr, pad in zip(arrs, pad_sizes)]
        return np.stack(x, axis=axis)

    def _read_array(self,
                    start: int,
                    stop: int,
                    channels: Sequence[int],
                    padvalue: float,
    ):
        """Reads samples between start & stop indices for each channel index
        in channels.

        Args:
            start:
                The start sample index to read.
            stop:
                The stop sample index to read (exclusive).
            channels:
                Sequence of channel indices to read from EDF.
            padvalue:
                Value to pad to channels that run out of data to return.
                Only applicable if sample rates of channels differ.

        Returns:
            A float64 2-D array of shape len(channels) x (stop-start).
        """

        # Locate record tuples that include start & stop samples for
        # each channel but only perform reads over unique record tuples.
        rec_tuples = self._find_records(start, stop, channels)
        uniq_tuples = set(rec_tuples)
        reads = {tup: self._records(*tup) for tup in uniq_tuples}

        result=[]
        for ch, rec_tup in zip(channels, rec_tuples):

            #get preread array and extract samples for this ch
            arr = reads[rec_tup]
            arr = arr[:, self.header.record_map[ch]].flatten()

            #adjust start & stop relative to records start pt
            a = start - rec_tup[0] * self.header.samples_per_record[ch]
            b = a + (stop - start)
            result.append(arr[a:b])

        res = self._padstack(result, padvalue)
        return self._decipher(res, channels)

    def read(self,
             start: int,
             stop: Optional[int] = None,
             padvalue: float = np.NaN
    ) -> npt.NDArray[np.float64]:
        """Reads samples from this EDF from this Reader's channels.

        Args:
            start:
                The start sample index to read.
            stop:
                The stop sample index to read (exclusive). If None, samples
                will be read until the end of file.
            padvalue:
                Value to pad to channels that run out of samples to return.
                Only applicable if sample rates of channels differ.

        Returns:
            A float64 array of shape len(chs) x (stop-start) samples.
        """

        if start > max(self.header.samples):
            return np.empty((len(self.channels), 0))

        if not stop:
            stop = max(self.header.samples)

        arr = self._read_array(start, stop, self.channels, padvalue)
        # use cast to indicate ndarray type for docs
        return cast(np.ndarray, arr)


# Writer groups logically related non-public methods.
# pylint: disable-next=too-few-public-methods
class Writer(bases.Writer):
    """A writer of European Data Format (EDF/EDF+) files.

    This Writer is a context manager for writing EEG data and metadata to an
    EDF binary file. Unlike Readers it must be opened under the context
    management protocol. Importantly, this writer does not currently support
    writing annotations to an EDF file.

    Attributes:
        path (Path):
            A python path instance to target file to write data to.

    Examples:
        >>> from openseize.demos import paths
        >>> filepath = paths.locate('recording_001.edf')
        >>> # Create a reader that will read only channels [0, 1]
        >>> # and write out these channels to a new file
        >>> writepath = paths.data_dir.joinpath('subset_001.edf')
        >>> with Reader(filepath) as reader:
        >>>     with Writer(writepath) as writer:
        >>>         writer.write(reader.header, reader, channels=[0, 1])
    """

    def __init__(self, path: Union[str, Path]) -> None:
        """Initialize this Writer. See base class for further details."""

        super().__init__(path, mode='wb')

    def _write_header(self, header: Header) -> None:
        """Writes a dict of EDF header metadata to this Writer's opened
        file.

        Args:
            header:
                A dict of EDF compliant metadata. Please see Header for
                further details.
        """

        # the header should be added during write not initialization
        # pylint: disable-next=attribute-defined-outside-init
        self.header = header
        bytemap = header.bytemap(header.num_signals)

        # Move to file start and write each ascii encoded byte string
        self._fobj.seek(0)
        for items, (nbytes, _) in zip(header.values(), bytemap.values()):
            items = [items] if not isinstance(items, list) else items

            for item, nbyte in zip(items, nbytes):
                bytestr = bytes(str(item), encoding='ascii').ljust(nbyte)
                self._fobj.write(bytestr)

    def _records(self,
                 data: Union[np.ndarray, Reader],
                 channels: Sequence[int]
    ) -> Generator[List[np.ndarray], None, None]:
        """Yields 1-D arrays, one per channel, to write to a data record.

        Args:
            data:
                An 2D array, memmap or Reader instance with samples along
                the last axis.
            channels:
                A sequence of channels to write to each data record.

        Yields:
            A list of single row 2-D arrays of samples for a single channel
            for a single data record.
        """

        for n in range(self.header.num_records):
            result = []
            # The number of samples per record is channel dependent if
            # sample rates are not equal across channels.
            starts = n * np.array(self.header.samples_per_record)
            stops = (n+1) * np.array(self.header.samples_per_record)

            for channel, start, stop in zip(channels, starts, stops):
                if isinstance(data, np.ndarray):
                    x = np.atleast_2d(data[channel][start:stop])
                    result.append(x)
                    #result.append(data[channel][start:stop])
                else:
                    data.channels = [channel]
                    result.append(data.read(start, stop))

            yield result

    def _encipher(self, arrs: Sequence):
        """Converts float arrays to 2-byte little-endian integer arrays
        using the EDF specification.

        Args:
            arrs:
                A sequence of 1-D arrays of float dtype.

        Returns:
            A sequence of 1-D arrays in 2-byte little-endian format.
        """

        slopes = self.header.slopes
        offsets = self.header.offsets
        results = []
        for ch, x in enumerate(arrs):
            arr = np.rint((x - offsets[ch]) / slopes[ch])
            arr = arr.astype('<i2')
            results.append(arr)
        return results

    def _validate(self, header: Header, data: np.ndarray) -> None:
        """Ensures the number of samples is divisible by the number of
        records.

        EDF files must have an integer number of records (i.e. the number of
        samples must fill the records). This may require appending 0's to
        the end of data to ensure this.

        Args:
            header:
                A Header instance of EDF metadata.
            data:
                The 2-D data with samples along the last axis.

        Raises:
            A ValueError if not divisible.
        """

        if data.shape[1] % header.num_records != 0:
            msg=('Number of data samples must be divisible by '
                 'the number of records; {} % {} != 0')
            msg.format(data.shape[1], header.num_records)
            raise ValueError(msg)

    def _progress(self, record_idx: int) -> None:
        """Relays write progress during file writing."""

        msg = 'Writing data: {:.1f}% complete'
        perc = record_idx / self.header.num_records * 100
        print(msg.format(perc), end='\r', flush=True)

    # override of general abstract method requires setting specific args
    # pylint: disable-next=arguments-differ
    def write(self,
              header: Header,
              data: Union[np.ndarray, Reader],
              channels: Sequence[int],
              verbose: bool = True,
              ) -> None:
        """Write header metadata and data for channel in channels to this
        Writer's file instance.

        Args:
            header:
                A mapping of EDF compliant fields and values.
            data:
                An array with shape (channels, samples) or Reader instance.
            channels:
                Channel indices to write to this Writer's open file.
            verbose:
                An option to print progress of write.

        Raises:
            ValueErrror: An error occurs if samples to be written is not
                         divisible by the number of records in the Header
                         instance.
        """

        header = Header.from_dict(header)
        header = header.filter(channels)
        self._validate(header, data)

        self._write_header(header) #and store header to instance
        self._fobj.seek(header.header_bytes)
        for idx, record in enumerate(self._records(data, channels)):
            samples = self._encipher(record) # floats to '<i2'
            samples = np.concatenate(samples, axis=1)
            #concatenate data bytes to str and write
            byte_str = samples.tobytes()
            self._fobj.write(byte_str)
            if verbose:
                self._progress(idx)


def splitter(path: Path,
             mapping: Dict,
             outdir: Optional[Union[str, Path]] = None,
) -> None:
    """Creates separate EDFs from a multichannel EDF.

    This tool is useful for partitioning an EDF with channels from different
    subjects into multiple single subject EDFs. The original 'unsplit' EDF
    is left unmodified.

    Args:
        path:
            Path to an EDF file to be split.
        mapping:
            A mapping of filenames and channel indices for each disjoined
            EDF file to be written.
        outdir:
            Directory where each file in mapping should be written. If None
            provided, the directory of the unsplit edf path will be used.
    """

    reader = Reader(path)
    outdir = Path(outdir) if outdir else reader.path.parent
    for fname, indices in mapping.items():
        target = outdir.joinpath(Path(fname).with_suffix('.edf'))
        with Writer(target) as outfile:
            outfile.write(reader.header, reader, indices)
    reader.close()
