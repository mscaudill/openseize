import numpy as np
from pathlib import Path

from openseize.io.headers import EDFHeader
from openseize.mixins import ViewInstance

def open_edf(path, mode):
    """Opens an edf file at path for reading or writing. 

    Args:
        path (Path):            Path instance to edf file
        mode (str):             one of the binary modes ('rb' for 
                                reading & 'wb' for writing, *+ modes 
                                not supported)
    
    Returns a Reader or Writer for this EDF file.
    """
    
    path = Path(path)
    if mode == 'rb':
        return Reader(path)
    elif mode == 'wb':
        return Writer(path)
    else:
        raise ValueError("invalid mode '{}'".format(mode))


class FileManager(ViewInstance):
    """A context manager for ensuring files are closed at the conclusion
    of reading or writing or in case of any raised exceptions.

    This class defines a partial interface and can not be instantiated.
    """

    def __init__(self, path, mode):
        """Initialize this FileManager inheritor with a path and mode."""

        if type(self) is FileManager:
            msg = '{} class cannot be instantiated'
            print('msg'.format(type(self).__name__))
        self.path = Path(path)
        self._fobj = open(path, mode)

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


class Reader(FileManager):
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
        self.header = EDFHeader(path)

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

    def _read_gen(self, start, channels, chunksize, padvalue):
        """A generator yielding arrays of data from an EDF file.

        Args:
            start (int):            start sample of returned data
            channels (list):        list of channels to return data for
            chunksize (int):        number of samples to return per iter
            padvalue (float):       value to pad to channels that run out of
                                    samples to return (see _padstack).
                                    Ignored if all channels have the same
                                    sample rates.
          
        Yields: arrays of shape chs x samples with float64 dtype
        """

        starts = range(start, max(self.header.samples), chunksize)
        for start, stop in zip(starts, starts[1:]):
            yield self._read_array(start, stop, channels, padvalue)

    def read(self, start, stop=None, channels=None, chunksize=30e6,
             padvalue=np.NaN):
        """Reads samples from an edf file for the specified channels.

        Depending on supplied arguments this function will return an array
        of samples or a generator of arrays of samples.

        Args:
            start (int):            start sample to read
            stop (int):             stop sample to read. If None return
                                    a generator that sequentially yields 
                                    arrays of shape (channels,chunksize)
                                    until end of file
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

        Returns/Yields:
            if stop is None, read yields arrays of chs x chunksize samples
            if stop is int, returns an array of chs x (stop-start) samples 
            array(s) yielded or returned are of dtype float64.
        """

        if start > max(self.header.samples):
            msg = 'start index {} is out of bounds for EDF with {} samples'
            raise EOFError(msg.format(start, max(self.header.samples)))
        #use all channels if None and dispatch to read method
        channels = self.header.channels if not channels else channels
        if not stop:
            #return generator
            return self._read_gen(start, channels, int(chunksize), padvalue)
        else:
            #return an array
            return self._read_array(start, stop, channels, padvalue)


class Writer(FileManager):
    """A writer for EDF file header & data records.

    This writer does not currently support writing annotations to an EDF's
    data records.
    """

    def __init__(self, path):
        """Initialize this Writer with a write path."""

        super().__init__(path, mode='wb')

    def _write_header(self, header):
        """Write header dict to header section of this Writer's fileobj.

        Args:
            header (dict):          a dict to use as this files header
        """
       
        #build EDFheader & convert all values to list
        self.header = EDFHeader.from_dict(header)
        lsheader = {k: v if isinstance(v, list) else [v]
                   for k, v in self.header.items()}
        #build bytemap dict and move to file start byte
        bmap = self.header.bytemap(self.header.num_signals)
        self._fobj.seek(0)
        #encode each header list el & write within bytecnt bytes
        for ls, (cnts, _) in ((lsheader[k], bmap[k]) for k in bmap):
            b = [bytes(str(x), 'ascii').ljust(n) for x, n in zip(ls, cnts)]
            self._fobj.write(b''.join(b))

    def _records(self, data, channels):
        """A generator yielding list of samples one per channel for a single
        record extracted from data.

        Args:
            data (ndarray, reader):     an ndarray, memmap or reader
                                        instance with shape channels x 
                                        samples 
            channels (list):            list of channels to write

        Yields: list of arrays of samples one per channel that constitute
                a record
        """

        for n in range(self.header.num_records):
            result = []
            #compute the start and stop sample for each channel
            starts = n * np.array(self.header.samples_per_record)
            stops = (n+1) * np.array(self.header.samples_per_record)
            for channel, start, stop in zip(channels, starts, stops):
                #fetch from array or reader and yield
                if isinstance(data, np.ndarray):
                    result.append(data[channel][start:stop])
                else:
                    result.append(data.read(start, stop, [channel]))
            yield result

    def _encipher(self, arrs):
        """Transform each 1-D array in arrays from floats to an array
        of 2-byte little-endian integers.

        see: _decipher method of the EDFReader.
        """

        slopes = self.header.slopes
        offsets = self.header.offsets
        results = []
        for ch, x in enumerate(arrs):
            arr = np.rint((x - offsets[ch]) / slopes[ch])
            arr = arr.astype('<i2')
            results.append(arr)
        return results

    def _validate(self, header, data):
        """Validates that the number of samples to be written is divisible
        by the number of records in the header."""

        if data.shape[1] % header.num_records != 0:
            msg=('Number of data samples must be divisible by '
                 'the number of records; {} % {} != 0')
            raise ValueError(msg.format(values, num_records))

    def _progress(self, idx):
        """Relays write progress during file writing."""

        msg = 'Writing data: {:.1f}% complete'
        perc = idx / self.header.num_records * 100
        print(msg.format(perc), end='\r', flush=True) 

    def write(self, header, data, channels, verbose=True):
        """Writes the header and data to write path.

        Args:
            header (dict):              dict containing required items
                                        of an EDF header (see io.headers)
            data (ndarray, reader):     an ndarray, memmap or reader
                                        instance with shape channels x 
                                        samples 
            channels (list):            list of channels to write
            verbose (bool):             display write percentage complete
        """

        #build & write header
        header = EDFHeader.from_dict(header)
        #filter the header to include only channels
        header = header.filter('channels', channels)
        #before write validate
        self._validate(header, data)
        self._write_header(header)
        #Move to data records section, fetch and write records
        self._fobj.seek(header.header_bytes)
        for idx, record in enumerate(self._records(data, channels)):
            samples = self._encipher(record)
            samples = np.concatenate(samples)
            #convert to bytes and write to file obj
            byte_str = samples.tobytes()
            self._fobj.write(byte_str)
            if verbose:
                self._progress(idx)
        print('')


class Splitter(ViewInstance):
    """A tool for creating multiple EDFs using a subset of the data from
    a single EDF file.

    EDFs may contain channels corresponding to more than one subject. This
    tool allows clients to split these EDFs into multiple EDFs one per 
    subject in the unsplit EDF.

    Note: Does not overwrite the original EDF.
    """

    def __init__(self, path):
        """Initialize with path to EDF to be split & read its header."""

        self.read_path = Path(path)
        with open_edf(self.read_path, 'rb') as infile:
            self.header = infile.header

    def _channels(self, attr, values):
        """Returns channels of the header's attribute matching values."""

        return [getattr(self.header, attr).index(v) for v in values]

    def _createpath(self, idx):
        """Creates a default writepath from this Splitter's readpath by
        appending idx to the readpath stem.

        Returns: new write Path instance
        """
        
        p = self.read_path
        stem = p.stem + '_{}'.format(idx)
        return Path.joinpath(p.parent, stem).with_suffix(p.suffix)
        
    def _split(self, attr, values, writepath, **kwargs):
        """Writes a single EDF to writepath filtering by attr in header
        whose values match values.

        Args:
            attr (str):         named attr of filter splitter's header
                                (see io.headers.filter)
            values (list):      values to filter splitter's header
            writepath (Path):   write location of this split
            kwargs:             passed to writers write method

        Developer note: This method has been excised from the split method
        to allow for multiprocessing of file splitting in the future.
        """

        with open_edf(self.read_path, 'rb') as reader:
            #get channel indices corresponding to values
            channels = self._channels(attr, values)
            with open_edf(writepath, 'wb') as writer:
                writer.write(reader.header, reader, channels, **kwargs) 

    def split(self, by, groups, writepaths=None, **kwargs):
        """Creates multiple EDFs from the EDF at Splitter's readpath

        Args:
            by (str):           header field to split EDF data on (e.g.
                                channels or names or transducers etc.)
            groups (lists):     lists of list specifying values of the by
                                attribute to write to each split edf; e.g. 
                                if by='channels' and groups = [[1,3], [2,4]]
                                then channels [1,3] will be written to the
                                first edf and channels [2,4] will be written
                                to the second EDF. Any valid list of the edf
                                header can be used to split with (e.g.
                                names, transducers. etc..)
            writepaths (list):  list of Path instances specifying write
                                location of each split EDF. If None,
                                writepaths will be built from the readpath
                                by appending the file index to the readpath
                                (e.g. <readpath>_0.edf). If provided the
                                number of writepaths must match the number
                                of sublist in groups. (Default None)
            kwargs:             passed to Writer's write method.
        """

        if writepaths: 
            if len(writepaths) != len(groups):
                msg = ('Number of paths to write {} does not match ',
                        'number of EDFs to be written {}')
                raise ValueError(msg)
        else:
            writepaths = [None] * len(groups)
        for idx, (values, path) in enumerate(zip(groups, writepaths)):
            path = self._createpath(idx) if not path else path
            print('Writing File {} of {}'.format(idx+1, len(groups)))
            self._split(by, values, path, **kwargs)






        

if __name__ == '__main__':

    from scripting.spectrum.io.eeg import EEG
    import time

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    path2 = '/home/matt/python/nri/data/openseize/test_write.edf'
    
    """
    splitter = Splitter(path)
    splitter._split('names', ['EEG 4 SEx10', 'EEG 2 SEx10', 'EEG 3 SEx10'], None)
    """

    #reader = Reader(path)
    #header = reader.header
    #data = EEG(path)

    """
    with open_edf(path, 'rb') as infile:
        res = infile.read(0,1000)
    """

    """
    t0 = time.perf_counter()
    with open_edf(path2, 'wb') as outfile:
        outfile.write(header, reader, channels=[0,1,2,3])
    print('Written in {} s'.format(time.perf_counter() - t0))
    """

    """
    gen = reader.read(start=0, channels=[0,1,2,3], chunksize=int(30e6))
    arr = reader.read(start=0, stop=1, channels=[0,3])
    """

    splitter = Splitter(path2)
    splitter.split('channels', [[0,2], [1,3]])

