import numpy as np
import pprint
import copy
from inspect import getmembers

class Header(dict):
    """An extended dict for reading a binary file's header data.

    Extensions:
    1. dot '.' notation access to the underlying dict data
    2. possible computed attrs as properties (see EDFHeader)
    3. echo repr showing underlying data and computed properties

    This is a partial Header implementation (i.e. cannot be instantiated).
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

        try:
            return self[name]
        except:
            #explicitly raise error, needed for getmembers
            raise AttributeError

    def bytemap(self):
        """Returns a dict specifying the number of bytes to sequentially 
        read for each named section of the header & a type conversion.

        e.g. {'version': ([8], str), 'maxes': ([16]*4, float), ...}
        Specifies that version is stored in the first 8 bytes and should be
        type converted to a string. Maxes are stored starting at byte 8 
        (i.e. last position) as 4 16 byte values. Each value should be 
        converted to a float and stored to a list. All bytemaps of concrete
        headers should follow these conventions. 
        """
       
        raise NotImplementedError

    def read(self, encoding='ascii'):
        """Returns the entire header of a file as a dict.

        Args:
            encoding (str):         the file encoding used (Default is
                                    'ascii' encoding)
        """

        header = dict()
        #if no path return empty dict
        if not self.path:
            return header
        with open(self.path, 'rb') as fp:
            #loop over the bytemap, read and store the decoded values
            for name, (nbytes, dtype) in self.bytemap().items():
                res = [dtype(fp.read(n).strip().decode(encoding=encoding))
                       for n in nbytes]
                header[name] = res[0] if len(res) == 1 else res
        return header

    @classmethod
    def from_dict(cls, dic):
        """Creates a Header instance from a dict."""

        instance = cls(path=None)
        instance.update(dic)
        #verify supplied dict has correct fields
        if set(dic) == set(instance.bytemap(nsignals=0)):
            return instance
        else:
            msg='Missing keys required to create a header of type {}.'
            raise ValueError(msg.format(cls.__name__))

    def _isprop(self, attr):
        """Returns True if attr is a property."""

        return isinstance(attr, property)

    def __str__(self):
        """Overrides dict's print string to show accessible properties."""

        #get header properties and return  pprinter fmt str
        props = [k for k, v in getmembers(self.__class__, self._isprop)]
        pp = pprint.PrettyPrinter(sort_dicts=False, compact=True)
        props = {'Accessible Properties': props}
        return pp.pformat(self) + '\n\n' + pp.pformat(props)


class EDFHeader(Header):
    """A dict representation of an EDF Header."""

    def bytemap(self, nsignals=None):
        """Specifies the number of bytes to sequentially read for each field
        in an EDF header and dataype conversions to apply.

        Args:
            nsignals (int):         number of signals to instantiate this
                                    bytemap. If None (Default) read directly
                                    from the edf file. nsignals includes
                                    ordinary signals (aka channels) &
                                    annotation signals.

        The EDF file specification defining this bytemap can be found @
        https://www.edfplus.info/specs/edf.html
        """

        nsignals = self.count_signals() if nsignals is None else nsignals
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
                'names': ([16] * nsignals, str),
                'transducers': ([80] * nsignals, str),
                'physical_dim': ([8] * nsignals, str),
                'physical_min': ([8] * nsignals, float),
                'physical_max': ([8] * nsignals, float),
                'digital_min': ([8] * nsignals, float),
                'digital_max': ([8] * nsignals, float),
                'prefiltering': ([80] * nsignals, str),
                'samples_per_record': ([8] * nsignals, int),
                'reserved_1': ([32] * nsignals, str)}

    def count_signals(self):
        """Returns the number of signals in this EDF."""

        with open(self.path, 'rb') as fp:
            fp.seek(252)
            return int(fp.read(4).strip().decode())

    @property
    def annotated(self):
        """Returns True if this is EDF contains annotations."""

        return True if 'EDF Annotations' in self.names else False
        
    @property
    def annotation(self):
        """Returns annotations signal index if present & None otherwise."""
        
        result = None
        if self.annotated:
            result = self.names.index('EDF Annotations') 
        return result

    @property
    def channels(self):
        """Returns the 'ordinary signal' indices."""

        signals = list(range(self.num_signals))
        if self.annotated:
            signals.pop(self.annotation)
        return signals

    @property
    def samples(self):
        """Returns summed sample count across records for each channel.

        The last record of the EDF may not be completely filled with
        recorded signal values depending on the software that created it.
        """

        nrecs = self.num_records
        samples = np.array(self.samples_per_record) * nrecs
        return [samples[ch] for ch in self.channels]

    @property
    def record_map(self):
        """Returns a list of slice objects for each signal in a record."""

        scnts = np.insert(self.samples_per_record,0,0)
        cum = np.cumsum(scnts)
        return list(slice(a, b) for (a, b) in zip(cum, cum[1:]))

    @property
    def slopes(self):
        """Returns the slope (gain) of each channel in this Header.

        The physical values p are linearly mapped from the digital values d:
        p = slope * d + offset, where the slope = 
        (pmax -pmin) / (dmax - dmin) & offset = p - slope * d for any (p,d)

        see also offsets property.
        Returns: an 1-D array of slope values one per signal in EDF.
        """

        pmaxes = np.array(self.physical_max)[self.channels]
        pmins = np.array(self.physical_min)[self.channels]
        dmaxes = np.array(self.digital_max)[self.channels]
        dmins = np.array(self.digital_min)[self.channels]
        return (pmaxes - pmins) / (dmaxes - dmins)

    @property
    def offsets(self):
        """Returns the offset of each channel in this Header.

        see slopes property
        Returns: an 1-D array of offset values one per signal in EDF.
        """

        pmins = np.array(self.physical_min)[self.channels]
        dmins = np.array(self.digital_min)[self.channels]
        return (pmins - self.slopes * dmins)
    
    def filter(self, by, values):
        """Returns a new EDF header instance filtered on the by attr that
        contains values.

        E.g. if by=names and values=['EEG1', 'EEG4'] the returned header
        will contain information for channel names mathching values. 

        Returns: new header instance
        """
        
        header = copy.deepcopy(self)
        #handle non-list values
        if not isinstance(values, list):
            return header if header[by] == values else dict()
        else:
            indices = [i for i, v in enumerate(getattr(header, by))
                       if v in values]
            for key, vals in header.items():
                if isinstance(vals, list):
                    header[key] = [vals[idx] for idx in indices]
            #match num_signals to filter len and return header
            header['num_signals'] = len(values)
            return header



        

if __name__ == '__main__':

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'

    #Test construction from path
    header = EDFHeader(path)
    #Test alternate constructor
    #header2 = EDFHeader.from_dict(header)
    #Test alternate constructor validation
    #header2.pop('transducers')
    #header3 = EDFHeader.from_dict(header2)
