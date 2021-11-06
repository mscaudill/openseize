import abc
import itertools
from pathlib import Path

import numpy as np

from openseize.types import mixins, producer
from openseize.io import headers

class Reader(abc.ABC, mixins.ViewInstance):
    """An ABC that defines all subclasses as context managers and describes
    required methods and properties."""

    def __init__(self, path, mode):
        """Initialize readers with a path & a read mode ('r' or 'rb')."""

        self.path = Path(path)
        self._fobj = open(path, mode)

    @abc.abstractmethod
    def read(self, start, stop, channels, **kwargs):
        """Returns an ndarray of shape channels x (stop-start)."""

    @abc.abstractmethod
    def as_producer(self, channels, chunksize, **kwargs):
        """Returns an iterable of arrays of shape chs x chunksize."""

    def __enter__(self):
        """Return reader instance as target variable of this context."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close this reader instance's file obj. & propagate errors by 
        returning None."""

        self.close()

    def close(self):
        """Close this reader instance's opened file object."""

        self._fobj.close()


class EDF(Reader):
    """A reader of European Data Format (EDF/EDF+) files.

    The EDF specification has a header section followed by data records
    Each data record contains all signals stored sequentially. EDF+
    files include an annotation signal within each data record. To
    distinguish these signals we refer to data containing signals as
    channels and annotation signals as annotation. For details on the EDF/+
    file specification please see:

    https://www.edfplus.info/specs/index.html

    Currently, this reader does not support the reading of annotation
    signals.
    """

    def __init__(self, path):
        """Initialize with a path, & construct header & file object."""

        super().__init__(path, mode='rb')
        self.header = headers.EDFHeader(path)

    @property
    def shape(self):
        """Returns the number of channels x number of samples."""

        return len(self.header.channels), max(self.header.samples)

    def _decipher(self, arr, channels, axis=-1):
        """Deciphers an array of integers read from an EDF into an array of
        voltage float values.

        Args:
            arr (ndarry):           2D-array of int type
            channels (list):        list of channels to decipher
            axis (int):             sample axis of arr

        The physical values p are linearly mapped from the digital values d:
        p = slope * d + offset, where the slope = 
        (pmax -pmin) / (dmax - dmin) & offset = p - slope * d for any (p,d)

        Returns: ndarray with shape matching input shape & float64 dtype
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
        """Returns tuples (one per signal) of start, stop record numbers
        that include the start, stop sample numbers

        Args:
            start (int):                start of sample range to read
            stop (int):                 stop of sample range to read
            channels (list):            list of channels to return record
                                        numbers for
        """

        spr = np.array(self.header.samples_per_record)[channels]
        starts = start // spr
        stops = np.ceil(stop / spr).astype('int')
        return list(zip(starts, stops))

    def _records(self, a, b):
        """Reads samples between the ath to bth record.

        Returns: a 2D array of shape (b-a) x sum(samples_per_record)
        
        Note 1:   Returns samples upto end of file if b > num. of records
        Note 2:   If the start record is off the file we return an empty 
                  array of shape (1,0)
        """

        #if the start record is off the file return an empty array
        if a >= self.header.num_records:
            return np.empty((1,0))
        #move to file start
        self._fobj.seek(0)
        #ensure last record is not off file and cnt records
        b = min(b, self.header.num_records)
        cnt = b - a
        #each sample is represented as a 2-byte integer
        bytes_per_record = sum(self.header.samples_per_record) * 2
        #get offset in bytes & num samples spanning a to b
        offset = self.header.header_bytes + a * bytes_per_record
        nsamples = cnt * sum(self.header.samples_per_record)
        recs = np.fromfile(self._fobj, '<i2', nsamples, offset=offset)
        #reshape the records to num_records x sum samples_per_rec
        arr = recs.reshape(cnt, sum(self.header.samples_per_record))
        return arr

    def _padstack(self, arrs, padvalue):
        """Pads 1-D arrays so that all lengths match and stacks them.

        Args:
            padvalue (float):          value to pad 

        The channels in the edf may have different sample rates. If
        a channel runs out of values to return, we pad that channel with
        padvalue to ensure the reader can return a 2-D array.

        Returns: a channels x samples array
        """

        req = max(len(arr) for arr in arrs)
        amts = [req - len(arr) for arr in arrs]
        if all(amt == 0 for amt in amts):
            return np.stack(arrs, axis=0)
        else:
            #convert to float for unlimited value pad
            x = [np.pad(arr.astype(float), (0, amt), 
                 constant_values=padvalue) for arr, amt in zip(arrs, amts)]
            return np.stack(x, axis=0)

    def _read_array(self, start, stop, channels, padvalue):
        """Returns samples from start to stop for channels of this EDF.

        Args:
            start (int):            start sample to begin reading
            stop (int):             stop sample to end reading (exclusive)
            channels (list):        channels to return samples for
            padvalue (float):       value to pad to channels that run out of
                                    samples to return (see _padstack).
                                    Ignored if all channels have the same
                                    sample rates.

        Returns: array of shape chs x samples with float64 dtype
        
        Note: Returns an empty array if start exceeds samples in file 
        as np.fromfile gracefully handles this for us
        """

        #locate record endpts for each channel
        rec_pts = self._find_records(start, stop, channels)
        #read the data for each unique record endpt tuple
        uniq_pts = set(rec_pts)
        reads = {pts: self._records(*pts) for pts in uniq_pts}
        #perform final slicing and transform for each channel
        result=[]
        for ch, pts in zip(channels, rec_pts):
            #get preread array and extract samples for this ch
            arr = reads[pts]
            arr = arr[:, self.header.record_map[ch]].flatten()
            #adjust start & stop relative to records start pt
            a = start - pts[0] * self.header.samples_per_record[ch]
            b = a + (stop - start)
            result.append(arr[a:b])
        res = self._padstack(result, padvalue)
        #decipher and return
        return self._decipher(res, channels)
    
    def read(self, start, stop=None, channels=None, padvalue=np.NaN):
        """Reads samples from an edf file for the specified channels.

        Args:
            start (int):            start sample to read
            stop (int):             stop sample to read. If None read all
                                    samples to end of file.
            channels (list):        indices to return or yield data from
                                    (Default None returns data on all
                                    channels in EDF)
            chunksize (int):        number of samples to return from
                                    generator per iteration (Default is
                                    30e6 samples). This value is ignored if
                                    stop sample is provided.
            padvalue (float):       value to pad to channels that run out of
                                    samples to return. Ignored if all 
                                    channels have the same sample rates.

        Returns: an array of chs x (stop-start) samples of dtype float64.
        """

        #use all channels if None
        channels = self.header.channels if not channels else channels
        #if out of samples return an empty array
        if start > max(self.header.samples):
            return np.empty((len(channels), 0))
        if not stop:
            stop = max(self.header.samples)
        #return an array
        return self._read_array(start, stop, channels, padvalue)

    def as_producer(self, chunksize, channels=None, padvalue=np.NaN):
        """Returns an iterable EDF instance that can be used to create an
        iterator of arrays of shape channels x chunksize.

        Args:
            channels (list):        indices to return or yield data from
                                    (Default None returns data on all
                                    channels in EDF)
            chunksize (int):        number of samples to return per batch
            padvalue (float):       value to pad to channels that run out of
                                    samples to return. Ignored if all 
                                    channels have the same sample rates.

        """

        return _ProduceFromEDF(self, chunksize, channels, padvalue=np.NaN)


class _ProduceFromEDF(producer.Producer, mixins.ViewInstance):
    """Producer yielding arrays of shape channels x chunksize from an EDF. 

    Openseize operations that are memory intensive may use this iterable to
    access EDF data in batches.

    Attrs:
        data:                       This is a Reader instance
        channels (list):            channels to include in each batch
        chunksize (int):            number of samples to yield per batch
        padvalue (float):           value to pad to channels that run out of
                                    samples to return. Ignored if all 
                                    channels have the same sample rates.
    """
    
    def __init__(self, data, chunksize, channels=None, padvalue=np.NaN):
        """Initialize this iterable."""

        #call Producer's init, channels & padval will be added as attrs
        super().__init__(data, chunksize, axis=-1, channels=channels,
                         padvalue=padvalue)
        #overwrite producer's channels attr if None passed
        if not channels:
            self.channels = self.data.header.channels

    @property
    def shape(self):
        """Return the shape of this iterable EDF."""

        return self.data.shape

    def __iter__(self):
        """Returns an iterator yielding arrays of shape channels x chunksize
        from a reader instance."""

        #make generators of start, stop samples & exhaust reader
        starts = itertools.count(start=0, step=self.chunksize)
        stops = itertools.count(start=self.chunksize, step=self.chunksize)
        for start, stop in zip(starts, stops): 
            arr = self.data.read(start, stop, self.channels, self.padvalue)
            #if exhausted close reader and exit
            if arr.size == 0:
                break
            yield arr

    def close(self):
        """Closes this iterable's file resource."""

        if hasattr(self.data, 'close'):
            self.data.close()
