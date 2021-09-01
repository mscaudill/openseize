import numpy as np

from openseize.io.headers import EDFHeader

def open_edf(path, mode):
    """Opens an edf file at path for reading or writing. 

    Args:
        path (Path):            Path to edf file
        mode (str):             one of the binary modes ('rb' for 
                                reading & 'wb' for writing, *+ modes 
                                not supported)
    
    Returns a Reader or Writer for this EDF file.
    """

    if mode == 'rb':
        return Reader(path)
    elif mode == 'wb':
        return Writer(path)
    else:
        raise ValueError("invalid mode '{}'".format(mode))


class FileManager:
    """A context manager for ensuring files are closed at the conclusion
    of reading or writing or in case of any raised exceptions.

    This class defines an interface and therefore can not be instantiated.
    """

    def __init__(self, path, mode):
        """Initialize this FileManager inheritor with a path and mode."""

        if type(self) is FileManager:
            msg = '{} class cannot be instantiated'
            print('msg'.format(type(self).__name__))
        self.path = path
        self.fobj = open(path, mode)

    def __enter__(self):
        """Return instance as target variable of this context."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close this instances file object & propagate any error by 
        returning None."""

        self.close()

    def close(self):
        """Close this instance's opened file object."""

        self.fobj.close()


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

    def _decipher(self, arr, axis=-1):
        """Deciphers an array of integers read from an EDF into an array of
        voltage float values.

        The physical values p are linearly mapped from the digital values d:
        p = slope * d + offset, where the slope = 
        (pmax -pmin) / (dmax - dmin) & offset = p - slope * d for any (p,d)

        Returns: ndarray with shape matching input shape & float64 dtype
        """

        slopes = self.header.slopes[self.header.channels]
        offsets = self.header.offsets[self.header.channels]
        #expand to 2-D for broadcasting
        slopes = np.expand_dims(slopes[self.header.channels], axis=axis)
        offsets = np.expand_dims(offsets[self.header.channels], axis=axis)
        result = arr * slopes
        result += offsets
        return result

    def _find_records(self, start, stop):
        """Returns tuples (one per signal) of start, stop record numbers
        that include the start, stop sample numbers

        Args:
            start (int):                start of sample range to read
            stop (int):                 stop of sample range to read
        """

        spr = np.array(self.header.samples_per_record)[self.header.channels]
        starts = start // spr
        stops = np.ceil(stop / spr).astype('int')
        return list(zip(starts, stops))

    def _records(self, a, b):
        """Reads all samples from the ath to bth record.


        Returns: a 1D array of length (b-a) * sum(samples_per_record)
        
        Note: Returns samples upto the end of the file if b exceeds the
        number of records as np.fromfile gracefully handles this for us
        """

        #move to file start
        self.fobj.seek(0)
        cnt = b - a
        #each sample is represented as a 2-byte integer
        bytes_per_record = sum(self.header.samples_per_record) * 2
        #return offset in bytes & num samples spanning a to b
        offset = self.header.header_bytes + a * bytes_per_record
        nsamples = cnt * sum(self.header.samples_per_record)
        return np.fromfile(self.fobj, dtype='<i2', count=nsamples, 
                           offset=offset)

    def read(self, start, stop):
        """Returns samples from start to stop for all channels of this EDF.

        Args:
            start (int):            start sample to begin reading
            stop (int):             stop sample to end reading (exclusive)

        Returns: array of shape chs x samples with float64 dtype
        
        Note: Returns an empty array if start exceeds samples in file 
        as np.fromfile gracefully handles this for us
        """

        #locate record ranges for chs & find unique ones to read
        recs = self._find_records(start, stop)
        urecs = set(recs)
        #read each unique record range
        reads = {urec: self._records(*urec) for urec in urecs}
        #reshape each read to num_records x summed samples per rec
        reads = {urec: arr.reshape(-1, sum(self.header.samples_per_record)) 
                 for urec, arr in reads.items()}
        #perform final slicing and transform for each channel
        result = []
        for idx, (ch, rec) in enumerate(zip(self.header.channels, recs)):
            #get preread array and slice out channel
            arr = reads[rec]
            arr = arr[:, self.header.record_map[ch]].flatten()
            #adjust start & stop relative to first read record & store
            a = start - rec[0] * self.header.samples_per_record[ch]
            b = a + (stop - start)
            result.append(arr[a:b])
        result = np.stack(result, axis=0)
        #decipher & return
        result = self._decipher(result)
        return result


class Writer(FileManager):
    """A writer for EDF file header & data records.

    This writer does not currently support writing annotations to an EDF's
    data records.
    """

    def __init__(self, path):
        """ """

        super().__init__(path, mode='wb')

    def _write_header(self, header):
        """Writes header values to the header of the file obj.

        Args:
            header (dict):          a dict to use as this files header
        """
       
        #add a header and convert all values to list
        self.header = EDFHeader.from_dict(header)
        d = {k: v if isinstance(v, list) else [v] 
             for k, v in self.header.items()}
        #build an edf bytemap dict and move to file start byte
        bmap = self.header.bytemap(self.header.num_signals)
        self.fobj.seek(0)
        #encode each value list in header and write
        for values, bytelist in ((d[name], bmap[name][0]) for name in bmap):
            bvalues = [bytes(str(value), encoding='ascii').ljust(nbytes) 
                      for value, nbytes in zip(values, bytelist)]
            #join the byte strings and write
            bstring = b''.join(bvalues)
            self.fobj.write(bstring)

    def _encipher(self, arr, axis=-1):
        """Transform arr of float values to an array of 2-byte 
        little-endian integers.

        see: _decipher method of the EDFReader.
        """

        slopes = self.header.slopes
        offsets = self.header.offsets
        #expand to 2-D for broadcasting
        slopes = np.expand_dims(slopes, axis=axis)
        offsets = np.expand_dims(offsets, axis=axis)
        #undo offset and gain and convert back to ints
        result = arr - offsets
        result = result / slopes
        #return rounded 2-byte little endian integers
        return np.rint(result).astype('<i2')

    def _write_record(self, arr, axis=-1):
        """Writes a single record of data to this writers file.

        Args:
            arr (ndarray):          a 1 or 2-D array of samples to write
            axis (int):             sample axis of the array
        """
       
        x = arr.T if axis == 0 else arr
        #encipher float array
        x = self._encipher(x)
        #slice each channel to handle different samples_per_rec
        for ch in self.header.channels:
            samples = x[ch, :self.header.samples_per_record[ch]]
            byte_str = samples.tobytes()
            self.fobj.write(byte_str)

    def _progress(self, idx):
        """Relays write progress during file writing."""

        msg = 'Writing data: {:.1f}% complete'
        perc = idx / self.header.num_records * 100
        print(msg.format(perc), end='\r', flush=True) 

    def _records(self, data, axis=-1):
        """Returns a generator of data record values to write.

        Args:
            data (array like):      an object supporting numpy's slicing
                                    protocol
            axis (int):             sample axis of data
        """

        cnt = max(self.header.samples_per_record)
        starts = [cnt * rec for rec in range(self.header.num_records+1)]
        endpts = zip(starts, starts[1:])
        slices = [[slice(None), slice(*pt)] for pt in endpts]
        if axis == 0:
            slices = [ls[-1::-1] for ls in slices]
        return (data[tuple(slc)] for slc in slices)

    def _validate(self, header, data):
        """Validates that number of header chanels matches number of data
        channels and samples to be written are evenly divisible by the
        number of records."""
 
        if len(header.channels) not in data.shape:
            msg = ('Number of channels in header does not match '
                    'the number of channels in the data')
            raise ValueError(msg)
        samples = data.shape[0] * data.shape[1]
        if samples % header.num_records != 0:
            msg=('Number of data samples must be divisible by '
                 'the number of records; {} % {} != 0')
            raise ValueError(msg.format(values, num_records))

    def write(self, header, data, axis=-1):
        """Writes the header and data to write path.

        Args:
            header (dict):              dict containing required items
                                        of an EDF header (see io.headers)
            data (ndarray, sequence):   a 2-D array-like obj returning
                                        samples for each channel along axis
                                        data is required to have a shape
                                        attr with sample shape along axis
            axis (int):                 sample axis of the data, must be one
                                        of (-1,0,1) as data is 2-D
        """

        #validate data has needed chs & can be split into num records
        self._validate(header, data)
        #build & write header
        header = EDFHeader.from_dict(header)
        self._write_header(header)
        #Move to data records section
        self.fobj.seek(header.header_bytes)
        for idx, record in enumerate(self._records(data, axis=axis)):
            self._progress(idx)
            self._write_record(record, axis=axis)
        

if __name__ == '__main__':

    from scripting.spectrum.io.eeg import EEG

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    path2 = '/home/matt/python/nri/data/openseize/test_write.edf'
    
    reader = Reader(path)
    header = reader.header
    data = EEG(path)

    """
    with open_edf(path, 'rb') as infile:
        res = infile.read(0,1000)
    """

    with open_edf(path2, 'wb') as outfile:
        outfile.write(header, data, axis=0)

    #fm = FileManager(path, 'rb')

