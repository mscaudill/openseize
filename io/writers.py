import numpy as np

from openseize.io import headers

class EDFWriter:
    """A context manager for writing a EDF header & data records."""

    def __init__(self, path, header):
        """Intialize this writer with a path."""
        
        self.header = headers.EDFHeader.from_dict(header)
        self.fobj = open(path, 'wb')
 
    def __enter__(self):
        """Return instance to the context target."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close file obj. & propogate any exceptions by returning None."""

        self.fobj.close()
  
    def write_header(self):
        """Writes header values to the header of the file obj."""
        
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

        The strored digital values (d) are linearly mapped from the physical
        values (p) via: d = (p - offset) / slope
        where the slope = (pmax -pmin) / (dmax - dmin) 
        & offset = p - slope * d for any (p,d)

        This undoes the transformation applied in the EDFReader.
        """

        slopes = self.header.slopes[self.header.channels]
        offsets = self.header.offsets[self.header.channels]
        #expand to 2-D for broadcasting
        slopes = np.expand_dims(slopes, axis=axis)
        offsets = np.expand_dims(offsets, axis=axis)
        #undo offset and gain and convert back to ints
        result = arr[channels] - offsets
        result = (result / slopes).astype(int)
        return result

    def write_record(self, arr, axis=-1):
        """axis will be sample axis """
       
        #FIXME
        # What is the incoming arrary? should it include only chs to write
        # or should it be sliced like below?
        # Are we handling possibly different sample rates correctly?

        if arr.ndim > 2:
            raise ValueError('array must be a 1 or 2-D array')
        x = arr.T if axis == 0 else arr
        x = self._detransform(x, axis=axis)
        for ch in self.header.channels:
            samples = x[ch][:self.header.samples_per_record[ch]]
            #write little endian 2-byte integers
            samples.tofile(self.fobj, '<i2')


    


if __name__ == '__main__':

    from openseize.io import headers

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    header = headers.EDFHeader(path)
    
    writepath = 'sandbox/test_write1.edf'
    #writer = EDFWriter('sandbox/test_write2.edf')
    #writer.write_header(header)

    arr = np.ones((5, 100)) * np.array([[0.0045, 0.009, 100, 0, 1]]).T
    with EDFWriter(writepath, header) as f:
        f.write_header()
        #res = f.write_record(arr)
