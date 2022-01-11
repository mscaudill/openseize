"""Context managers for writing EEG data to a variety of file types.

This module defines a collection Writer types for writing EEG files from
arrays and Reader like objects to EDF files. Each file type specific writer
is expected to inherit the Writer ABC and override its write abstract
method.

Typical usage example:

#create an EDF reader to an edf file stored on disk 
data = Readers.EDF(<*.edf path>)

#select a subset of channels
chs = [0,3]

#write a new EDF file to save path containing only a subset of original chs
with EDF(<save path>) as outfile:
    outfile.write(header, data, channels=chs)
"""

import abc
import numpy as np
from pathlib import Path

from openseize.io.headers import headers
from openseize.io.headers import bytemaps
from openseize.core import mixins


class Writer(abc.ABC, mixins.ViewInstance):
    """An ABC that defines all subclasses as context managers and describes
    required methods and properties."""

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


class EDFWriter(Writer):
    """A writer of European Data Format (EDF) files.

    This writer does not support writing annotations to an EDF file.
    """

    def __init__(self, path):
        """Initialize this writer. See base class for futher details."""

        super().__init__(path, mode='wb')

    def _write_header(self, header):
        """Write a header dict to this Writer's opened file instance.

        Args:
            header: dict          
                A dict of EDF compliant metadata. Please see 
                io.headers.EDFHeader for further details.
        """
       
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
        """

        self.header = header
        bytemap = bytemaps.edf(header.num_signals)
        # Move to file start and write each ascii encoded byte string
        self._fobj.seek(0)
        for items, (nbytes, _) in zip(header.values(), bytemap.values()):
            items = [items] if not isinstance(items, list) else items
            for item, nbyte in zip(items, nbytes):
                bytestr = bytes(str(item), encoding='ascii').ljust(nbyte)
                self._fobj.write(bytestr)

    def _records(self, data, channels):
        """Yields a list of sample arrays one per channel to write to
        a single data record

        Args:
            data: 2-D array, memmap, Reader instance
                An object that is sliceable or has a read method that 
                returns arrays of shape channels x samples. 
            channels: sequence
                A sequence of channels to write to each data record in this
                Writer.

        Yields:
            A list of 1-D arrays of samples for a single data record, one
            array per channel in channels.
        """

        for n in range(self.header.num_records):
            result = []
            # The number of samples per record is channel dependent if
            # sample rates are not equal across channels.
            starts = n * np.array(self.header.samples_per_record)
            stops = (n+1) * np.array(self.header.samples_per_record)

            for channel, start, stop in zip(channels, starts, stops):
                if isinstance(data, np.ndarray):
                    result.append(data[channel][start:stop])
                else:
                    result.append(data.read(start, stop, [channel]))
            
            yield result

    def _encipher(self, arrs):
        """Transforms each array in a sequence of arrays from float to
        a 2-byte little-endian integer dtype.

        Args:
            arrs: sequence
                Sequence of 1-D arrays of float dtype.

        See also _decipher method of the EDFReader.

        Returns:
            A sequence of 1-D arrays in 2-byte little-endian fmt.
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
        """Validates that samples in data is divisible by number of records
        in header.

        The EDF file spec. does not allow the writing of partial record at
        the end of the file. Therefore, the data to be written needs to 
        perfectly fill all num_records in the header. This may require 
        appending 0's to the end of data to ensure this.

        Args:
            header: dict
                An EDFHeader instance.
            data: 2-D array or Reader instance
                Data to be written to this Writer's open file instance.
        
        Raises:
            A ValueError if num. samples is not divisible by num. records 
            in header. 
        """

        if data.shape[1] % header.num_records != 0:
            msg=('Number of data samples must be divisible by '
                 'the number of records; {} % {} != 0')
            raise ValueError(msg.format(values, num_records))

    def _progress(self, record_idx):
        """Relays write progress during file writing."""
        
        msg = 'Writing data: {:.1f}% complete'
        perc = record_idx / self.header.num_records * 100
        print(msg.format(perc), end='\r', flush=True) 

    def write(self, header, data, channels, verbose=True):
        """Write header & data for each channel to file object.

        Args:
            header: dict
                A mapping of EDF compliant fields and values. For Further
                details see EDFHeader in io.headers module.
            data: 2-D array or Reader instance
                A channels x samples array or Reader instance.
            channels: sequence
                A sequence of channel indices to write to this Writer's 
                open file instance.
            verbose: bool
                An option to print progress of write. Default (True) prints
                status update as each record is written.
        """

        header = headers.EDFHeader.from_dict(header)
        header = header.filter(channels)
        self._validate(header, data)

        self._write_header(header)
        self._fobj.seek(header.header_bytes)
        for idx, record in enumerate(self._records(data, channels)):
            samples = self._encipher(record) # floats to '<i2'
            samples = np.concatenate(samples)
            #concatenate data bytes to str and write
            byte_str = samples.tobytes()
            self._fobj.write(byte_str)
            if verbose:
                self._progress(idx)


        

if __name__ == '__main__':

    import time
    from openseize.io import readers

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    path2 = '/home/matt/python/nri/data/openseize/test_write.edf'
    

    reader = readers.EDFReader(path)
    header = reader.header

    t0 = time.perf_counter()
    with EDFWriter(path2) as writer:
        writer.write(header, reader, channels=[0,1])
    print('Written in {} s'.format(time.perf_counter() - t0))


  
