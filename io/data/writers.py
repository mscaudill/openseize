import abc
import numpy as np
from pathlib import Path

from openseize.io.data.headers import EDFHeader
from openseize.mixins import ViewInstance

class Writer(abc.ABC, ViewInstance):
    """An ABC that defines all subclasses as context managers and describes
    required methods and properties."""

    def __init__(self, path, mode):
        """Initialize this Writer with a path & a write mode 'w' or 'wb'."""

        self.path = Path(path)
        self._fobj = open(path, mode)

    @abc.abstractmethod
    def write(self, header, data, channels, **kwargs):
        """Writes a header and data for channels to a file obj."""

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


class EDF(Writer):
    """A writer of European Data Format (EDF) files.

    This writer does not support writing annotations to an EDF file.
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


        

if __name__ == '__main__':

    import time
    from openseize.io.data import readers

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    path2 = '/home/matt/python/nri/data/openseize/test_write.edf'
    

    reader = readers.EDF(path)
    header = reader.header

    t0 = time.perf_counter()
    with EDF(path2) as writer:
        writer.write(header, reader, channels=[0,1,2,3])
    print('Written in {} s'.format(time.perf_counter() - t0))


  
