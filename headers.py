class Header(dict):
    """An extended dict for reading and storing header data.

    This is a partial Header implementation thus it cannot be instantiated.
    Concrete headers are expected to define a 'bytemap' method to complete 
    the implementation.
    """

    def __init__(self, path):
        """Initialize this Header with a pathlib Path obj."""

        self.path = path
        #call dict's initializer & update with header data
        dict.__init__(self)
        self.update(self.read())

    def __getattr__(self, name):
        """Provides '.' notation access to this Header's values."""

        return self[name]

    def bytemap(self):
        """Returns a dict specifying the number of bytes to sequentially 
        read for each named section of the header & a type conversion.

        e.g. {'version': ([8], str), 'maxes': ([16]*4, float), ...}
        Specifies that version is stored in the first 8 bytes and should be
        type converted to a UTF-8 string. Bytemap then specifies that maxes
        are stored starting at byte 8 (last position) as 4 16 byte values
        and that each value should be converted to a float and stored to
        a list. 
        """
       
        raise NotImplementedError

    def read(self):
        """Returns the entire header of a file as a dict."""

        header = dict()
        with open(self.path, 'rb') as fp:
            #loop over the bytemap, read and store the decoded values
            for name, (nbytes, dtype) in self.bytemap().items():
                res = [dtype(fp.read(n).strip().decode()) for n in nbytes]
                header[name] = res[0] if len(res) == 1 else res
        return header


class EDFHeader(Header):
    """A dict representation of an EDF Header."""

    def bytemap(self):
        """Specifies the number of bytes to sequentially read for each field
        in an EDF header.

        The EDF file specification defining this bytemap can be found @
        https://www.edfplus.info/specs/edf.html
        """

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
                'names': ([16] * self.num_signals, str),
                'transducers': ([80] * self.num_signals, str),
                'physical_dim': ([8] * self.num_signals, str),
                'physical_min': ([8] * self.num_signals, float),
                'physical_max': ([8] * self.num_signals, float),
                'digital_min': ([8] * self.num_signals, float),
                'digital_max': ([8] * self.num_signals, float),
                'prefiltering': ([80] * self.num_signals, str),
                'samples_per_record': ([8] * self.num_signals, int),
                'reserved_1': ([32] * self.num_signals, str)}

    @property
    def num_signals(self):
        """Returns the number of signals in this EDF."""

        with open(self.path, 'rb') as fp:
            fp.seek(252)
            return int(fp.read(4).strip().decode())

        

if __name__ == '__main__':

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'

    header = EDFHeader(path)
