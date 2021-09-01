import numpy as np

from openseize.io import headers

class EDFWriter:
    """A context manager for writing a EDF header & data records."""

    def __init__(self, path):
        """Intialize this writer with a path."""
        
        self.fobj = open(path, 'wb')
 
    def __enter__(self):
        """Return instance to the context target."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close file obj. & propogate any exceptions by returning None."""

        self.fobj.close()
  
    def _write_header(self, header):
        """Writes header values to the header of the file obj."""
        
        self.header = headers.EDFHeader.from_dict(header)
        #convert all values in header to list for consistency
        d = {k: v if isinstance(v, list) else [v] 
             for k, v in self.header.items()}
        bmap = self.header.bytemap(self.header.num_signals)
        #encode each value in values and justify with empty chr
        self.fobj.seek(0)
        for values, bytelist in ((d[name], bmap[name][0]) for name in bmap):
            bvalues = [bytes(str(value), encoding='ascii').ljust(nbytes) 
                      for value, nbytes in zip(values, bytelist)]
            #join the byte strings and write
            bstring = b''.join(bvalues)
            self.fobj.write(bstring)

    def _detransform(self, arr, axis=-1):
        """Linearly rransforms an arr of float values to integers.

        see: transform method of the EDFReader.
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

    def _write_annotation(self):
        """Writes a single records annotations."""

        allocated = self.header.samples_per_record[self.header.annotation]
        annotation = np.zeros(allocated)
        #annotation byte order is unchanged (i.e. big-endian)
        #annotation.tofile(self.fobj, sep="", 'i2')
    
    def _write_record(self, arr, axis=-1):
        """Writes a single record of data to this writers file.

        Args:
            arr (ndarray):          a 1 or 2-D array of samples to write
            axis (int):             sample axis of the array

        The arr must contain data for channels specified in the header
        and must contain enough samples to write.
        e.g if samples_per_record = [50k, 50k, 10k] the size of arr must be
        3 channels x 50k samples for each channel. The channel with 10k 
        samples will be sliced to the correct num of samples before writing.
        """
       
        x = arr.T if axis == 0 else arr
        x = self._detransform(x, axis=axis)
        for ch in self.header.channels:
            samples = x[ch, :self.header.samples_per_record[ch]]
            #print('Integer samples: {}'.format(samples[:10]))
            #write little endian 2-byte integers
            #for some reason these methods dont agree!!
            #samples.tofile(self.fobj, sep="", format='<i2')
            byte_str = samples.tobytes()
            self.fobj.write(byte_str)
            #write the annotations 
            #if self.header.annotated:
            #    self._write_annotation()

    def write(self, header, data, channels, axis=-1):
        """

        """

        if all(isinstance(ch, str) for ch in channels):
            channels = [self.header.names.index[ch] for ch in channels]
        fheader = header.filter(by='channels', values=channels)
        self._write_header(fheader)
        samples = max(fheader.samples_per_record)
        starts = [samples * x for x in range(fheader.num_records)]
        slc = [slice(None)] * 2
        #FIXME I don't appear to go to end of data
        #FIXME by writing multiple records at a time
        self.fobj.seek(self.header.header_bytes)
        for idx, (start, stop) in enumerate(zip(starts, starts[1:])):
            #if idx > 0:
            #    break
            print('Record # {}'.format(idx), end='\r', flush=True)
            slc[axis] = slice(start, stop)
            #print(slc, end='\r', flush=True)
            arr = data[tuple(slc)]
            arr = arr.T if axis==0 else arr
            arr = arr[channels]
            #print('Input arr values: {}'.format(arr[:,:10]))
            self._write_record(arr, axis=-1)
        return fheader



    


    


if __name__ == '__main__':

    from openseize.io import headers
    from scripting.spectrum.io.eeg import EEG

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    writepath = '/home/matt/python/nri/data/openseize/test_write.edf'
    header = headers.EDFHeader(path)
    data = EEG(path)

    with EDFWriter(writepath) as f:
        header = f.write(header, data, channels=[0,1,2,3], axis=0)
