import numpy as np

from openseize.io import headers

class EDFWriter:
    """A context manager for writing a EDF header & data records."""

    def __init__(self, path, header):
        """Intialize this writer with a path."""
        
        self.header = headers.EDFHeader.from_dict(header)
        self.fobj = open(path)
 
    def __enter__(self):
        """Return instance to the context target."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close file obj. & propogate any exceptions by returning None."""

        self.fobj.close()
  
    @property
    def bytemap(self):
        """Returns a dict describing the number of bytes to write for each
        header value or header list value."""

        nsigs = self.header.num_signals
        return {'version': ([8]), 
                'patient': ([80]), 
                'recording': ([80]),
                'start_date': ([8]),
                'start_time': ([8]),
                'header_bytes': ([8]),
                'reserved_0': ([44]),
                'num_records': ([8]),
                'record_duration': ([8]),
                'num_signals': ([4]),
                'names': ([16] * nsigs),
                'transducers': ([80] * nsigs),
                'physical_dim': ([8] * nsigs),
                'physical_min': ([8] * nsigs),
                'physical_max': ([8] * nsigs),
                'digital_min': ([8] * nsigs),
                'digital_max': ([8] * nsigs),
                'prefiltering': ([80] * nsigs),
                'samples_per_record': ([8] * nsigs),
                'reserved_1': ([32] * nsigs)}

    def write_header(self):
        """Writes header values to the header of the file obj."""
        
        #convert all values in header to list for consistency
        d = {k: v if isinstance(v, list) else [v] 
             for k, v in self.header.items()}
        bmap = self.bytemap
        #encode each value in values and justify with empty chr
        self.fobj.seek(0)
        for values, bytelist in ((d[name], bmap[name]) for name in bmap):
            bvalues = [bytes(str(value), encoding='ascii').ljust(nbytes) 
                      for value, nbytes in zip(values, bytelist)]
            #join the byte strings and write
            bstring = b''.join(bvalues)
            self.fobj.write(bstring)

    def _untransform(self, arr, axis=-1):
        """ """

        channels = list(range(self.header.num_signals))
        if 'EDF Annotations' in header.names:
            channels.pop(self.header.names.index('EDF Annotations'))
        pmaxes = np.array(self.header.physical_max)
        pmins = np.array(self.header.physical_min)
        dmaxes = np.array(self.header.digital_max)
        dmins = np.array(self.header.digital_min)
        slopes = (pmaxes - pmins) / (dmaxes - dmins)
        offsets = (pmins - slopes * dmins)
        #expand to 2-D for broadcasting
        slopes = np.expand_dims(slopes[channels], axis=axis)
        offsets = np.expand_dims(offsets[channels], axis=axis)
        #undo offset and gain and convert back to ints
        result = arr[channels] - offsets
        result = (result / slopes).astype(int)
        return result

    def write_record(self, arr, axis=-1):
        """axis will be sample axis """
        #FIXME should I have a record number here? Probably not
        if arr.ndim > 2:
            raise ValueError('array must be a 1 or 2-D array')
        x = arr.T if axis == 0 else arr
        x = self._untransform(x, axis=axis)
        #need to reshape
        x = x.flatten()
        #write little endian 2-byte integers
        #x.tofile(self.fobj, '<i2')
        print(x)


    


if __name__ == '__main__':

    from openseize.io import headers

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    header = headers.EDFHeader(path)
    
    writepath = 'sandbox/test_write1.edf'
    #writer = EDFWriter('sandbox/test_write2.edf')
    #writer.write_header(header)

    arr = np.ones((5, 100)) * np.array([[0.0045, 0.009, 100, 0, 1]]).T
    with EDFWriter(writepath, header) as f:
        res = f.write_record(arr)
