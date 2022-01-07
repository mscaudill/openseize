"""Context managers for reading EEG data from a variety of file types.

This module defines a collection of Reader types for reading EEG files.
Formats currently supported include: EDF. Each file type specific reader is
expected to inherit the Reader ABC and override its abstract read method.

Typical usage example:
   
# Read samples 10 to 1000 for channels 0, 1 & 3 to a 'data' numpy array.
# File will be closed automatically.

with EDf(*.edf) as infile:
    data = infile.read(10, 1000, channels=[0,1,3])

# Read samples 10 to 1000 for channels 0 & 2 to a 'data' numpy array.
# File must closed by calling 'close' method.

eeg = EDF(*.edf)
data = eeg.read(10, 1000, [0, 2])
eeg.close()
"""

import abc
import numpy as np

from pathlib import Path
from openseize.core import mixins
from openseize.io import headers


class Reader(abc.ABC, mixins.ViewInstance):
    """An ABC that defines all subclasses as context managers and describes
    required methods and properties."""

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

    @abc.abstractmethod
    def read(self, start, stop, channels, **kwargs):
        """Returns a numpy array of sample values between start and stop for
        each channel in channels.

        Args:
            start: int
                Start sample index of file read.
            stop: int
                Stop sample index of file read (exclusive).
            channels: sequence
                Sequence of channel indices to read.
            
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


class EDFReader(Reader):
    """A reader of European Data Format (EDF/EDF+) files.

    The EDF specification has a header section followed by data records
    Each data record contains all signals stored sequentially. EDF+
    files include an annotation signal within each data record. To
    distinguish these signals we refer to data containing signals as
    channels and annotation signals as annotation. Currently, this reader
    does not support the reading of annotation signals.

    For details on the EDF/+ file specification please see:

    https://www.edfplus.info/specs/index.html

    Attributes:
        header: A dictionary representation of an EDF Header.
        shape: A tuple of channels, samples contained in this EDF
    """

    def __init__(self, path):
        """Extends the Reader ABC with a header attribute."""

        super().__init__(path, mode='rb')
        self.header = headers.EDFHeader(path)

    @property
    def shape(self):
        """Returns a 2-tuple containing the number of channels and 
        number of samples in this EDF."""

        return len(self.header.channels), max(self.header.samples)

    def _decipher(self, arr, channels, axis=-1):
        """Converts an array of EDF integers to an array of voltage floats.

        The EDF file specification asserts that the physical voltage 
        values 'p' are linearly mapped from the integer digital values 'd'
        in the EDF according to:

            p = slope * d + offset
            slope = (pmax -pmin) / (dmax - dmin)
            offset = p - slope * d for any (p,d)

        The EDF header contains pmax, pmin, dmax and dmin for each channel.

        Args:
            arr: 2-D array
                An array of integer digital values read from the EDF file.
            channels: sequence
                Sequence of channels read from the EDF file
            axis: int
                The samples axis of arr. Default is last axis.
            
        Returns: 
            A float64 ndarray of physical voltage values with shape matching
            input 'arr' shape.
        """

        slopes = self.header.slopes[channels]
        offsets = self.header.offsets[channels]
        #expand to 2-D for broadcasting
        slopes = np.expand_dims(slopes, axis=axis)
        offsets = np.expand_dims(offsets, axis=axis)
        result = arr * slopes
        result += offsets
        return result

    def _find_records(self, start, stop, channels):
        """Locates a start and stop record number containing the start and
        stop sample number for each channel in channels.

        EDF files are partitioned into records. Each record contains data
        for each channel sequentially. Below is an example record for
        4-channels of data.

        Record:
        ******************************************************
        * Ch0 samples, Ch1 samples, Ch2 samples, Ch3 samples *
        ******************************************************

        The number of samples for each chanel will be different if the
        sample rates for the channels are not equal. The number of samples
        in a record for each channel is given in the header by the field
        samples_per_record. This method locates the start and stop record
        numbers that include the start and stop sample indices for each
        channel.

        Args:
            start: int
                The start sample to read.
            stop: int
                The stop sample to read (exclusive).
            channels: sequence
                Sequence of channels to read.

        Returns: 
            A list of 2-tuples containing the start and stop record numbers
            that inlcude the start and stop sample number for each channel
            in channels.
        """

        spr = np.array(self.header.samples_per_record)[channels]
        starts = start // spr
        stops = np.ceil(stop / spr).astype('int')
        return list(zip(starts, stops))

    def _records(self, a, b):
        """Reads samples between the ath to bth record.

        If b exceeds the number of records in the EDF, then samples upto the
        end of file are returned. If a exceeds the number of records, an
        empty array is returned.

        Args:
            a: int
                The start record to read.
            b: int
                The last record to be read (exclusive).

        Returns:
            A 2-D array of shape (b-a) x sum(samples_per_record)
        """

        if a >= self.header.num_records:
            return np.empty((1,0))
        b = min(b, self.header.num_records)
        cnt = b - a
        
        self._fobj.seek(0)
        #EDF samples are 2-byte integers
        bytes_per_record = sum(self.header.samples_per_record) * 2
        #get offset in bytes & num samples spanning a to b
        offset = self.header.header_bytes + a * bytes_per_record
        nsamples = cnt * sum(self.header.samples_per_record)
        #read records and reshape to num_records x sum(samples_per_record)
        recs = np.fromfile(self._fobj, '<i2', nsamples, offset=offset)
        arr = recs.reshape(cnt, sum(self.header.samples_per_record))
        return arr

    def _padstack(self, arrs, value, axis=0):
        """Pads a sequence of 1-D arrays to equal length and stacks them
        along axis.

        Args:
            arrs: sequence
                A sequence of 1-D arrays.
            value: float
                Value to append to arrays that are shorter than the longest
                array in arrs.
        
        Returns:
            A 2-D array.
        """

        longest = max(len(arr) for arr in arrs)
        pad_sizes = np.array([longest - len(arr) for arr in arrs])

        if all(pad_sizes == 0):
            return np.stack(arrs, axis=0)
        
        else:
            x = [np.pad(arr.astype(float), (0, pad), constant_values=value)
                    for arr, pad in zip(arrs, pad_sizes)]
            return np.stack(x, axis=0)

    def _read_array(self, start, stop, channels, padvalue):
        """Reads samples between start & stop for each channel in channels.

        Args:
            start: int
                The start sample index to read.
            stop: int
                The stop sample index to read (exclusive).
            channels: sequence
                Sequence of channels to read from EDF.
            padvalue: float
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
    
    def read(self, start, stop=None, channels=None, padvalue=np.NaN):
        """Reads samples from this EDF for the specified channels.

        Args:
            start: int
                The start sample index to read.
            stop: int
                The stop sample index to read (exclusive). If None, samples
                will be read until the end of file. Default is None.
            channels: sequence
                Sequence of channels to read from EDF. If None, all channels
                in the EDF will be read. Default is None.
            padvalue: float
                Value to pad to channels that run out of samples to return.
                Only applicable if sample rates of channels differ. Default
                padvalue is NaN.

        Returns: 
            A float64 array of shape len(chs) x (stop-start) samples.
        """

        channels = self.header.channels if not channels else channels
        if start > max(self.header.samples):
            return np.empty((len(channels), 0))
        if not stop:
            stop = max(self.header.samples)
        return self._read_array(start, stop, channels, padvalue)
