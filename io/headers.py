"""Extended dictionary for reading and storing EEG metadata.

This module contains a collection of Header types for storing metadata
from a variety of EEG file formats. Headers are dictionaries with the 
following additions:

1. A dot '.' notation dictionary key access to dictionary values.
2. An echo repr showing underlying data and format specific properties.

The EEG formats currently supported include: EDF. Each format specific header
is expected to inherit Header and override its bytemap method. Headers may be
alternately constructed from a dictionary. 

Typical usage example:

header = EDFHeader(path)

See also openseize.io.bytemaps for detailed information about byte locations
of metadata used to construct format specific Headers.
"""

import numpy as np
import pprint
import copy
from inspect import getmembers
from pathlib import Path

from openseize.io.headers import bytemaps


class Header(dict):
    """Base class for reading & storing an EEG file's header data.

    This base class defines the expected interface for all format specific
    headers. It is not instantiable. Inheriting classes must override the
    bytemap method.
    """

    def __init__(self, path):
        """Initialize this Header.

        Args:
            path: Path instance or str
                A path to eeg data file.
        """

        self.path = Path(path) if path else None
        dict.__init__(self)
        self.update(self.read())

    def __getattr__(self, name):
        """Provides '.' notation access to this Header's values.
        
        Args:
            name: str
                String name of a header bytemap key or header property.
        """

        try:
            return self[name]
        except:
            msg = "'{}' object has not attribute '{}'"
            raise AttributeError(msg.format(type(self).__name__, name))

    def bytemap(self):
        """Returns a format specific mapping specifying byte locations 
        in an eeg file containing header data. 

        Format specific bytemaps can be found in  opensieze.io.bytemaps.
        Please see this module for further details.
        """

        raise NotImplementedError

    def read(self, encoding='ascii'):
        """Reads the header of a file into a dict using a file format
        specific bytemap.

        Args:
            encoding: str         
                The encoding used to write the header data. Default is ascii
                encoding.
        """

        header = dict()
        if not self.path:
            return header

        with open(self.path, 'rb') as fp:
            # loop over the bytemap, read and store the decoded values
            for name, (nbytes, dtype) in self.bytemap().items():
                res = [dtype(fp.read(n).strip().decode(encoding=encoding))
                       for n in nbytes]
                header[name] = res[0] if len(res) == 1 else res
        return header

    @classmethod
    def from_dict(cls, dic):
        """Alternative constructor that creates a Header instance from
        a dictionary.

        Args:
            dic: dictionary 
                A dictionary containing all expected bytemap keys.
        """

        raise NotImplementedError

    def _isprop(self, attr):
        """Returns True if attr is a property of this Header.

        Args:
            attr: str
                The name of a Header attribute.
        """

        return isinstance(attr, property)

    def __str__(self):
        """Overrides dict's print string to show accessible properties."""

        # get header properties and return  pprinter fmt str
        props = [k for k, v in getmembers(self.__class__, self._isprop)]
        pp = pprint.PrettyPrinter(sort_dicts=False, compact=True)
        props = {'Accessible Properties': props}
        return pp.pformat(self) + '\n\n' + pp.pformat(props)


class EDFHeader(Header):
    """An extended dictionary representation of an EDF Header."""

    def bytemap(self):
        """Specifies the number of bytes to sequentially read for each field
        in an EDF header and dataype conversions to apply (see bytemaps
        module).

        The EDF file specification defining this bytemap can be found @
        https://www.edfplus.info/specs/edf.html
        """

        nsignals = self.count_signals()
        return bytemaps.edf(nsignals)

    def count_signals(self):
        """Returns the number of signals, including possible annotations,
        in this EDF."""

        with open(self.path, 'rb') as fp:
            fp.seek(252) # edf specifies num signals at 252nd byte
            return int(fp.read(4).strip().decode())

    @classmethod
    def from_dict(cls, dic):
        """Alternative constructor that creates a Header instance from
        a dictionary.

        Args:
            dic: dictionary 
                A dictionary containing all expected bytemap keys.
        """

        instance = cls(path=None)
        instance.update(dic)
        # validate dic contains all bytemap keys
        if set(dic) == set(bytemaps.edf(1)):
            return instance
        else:
            msg='Missing keys required to create a header of type {}.'
            raise ValueError(msg.format(cls.__name__))

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
        """Returns a list of slices corresponding to the start, stop sample
        indices within a record for each channel.

        Within a record each channel is sequentially listed like so:

        Record:
        ******************************************************
        * Ch0 samples, Ch1 samples, Ch2 samples, Ch3 samples *
        ******************************************************

        This function returns the start, stop indices for each channel as
        a list of slices.
        """

        scnts = np.insert(self.samples_per_record,0,0)
        cum = np.cumsum(scnts)
        return list(slice(a, b) for (a, b) in zip(cum, cum[1:]))

    @property
    def slopes(self):
        """Returns the slope (gain) of each channel in this Header.

        The EDF file specification asserts that the physical voltage 
        values 'p' are linearly mapped from the integer digital values 'd'
        in the EDF according to:

            p = slope * d + offset
            slope = (pmax -pmin) / (dmax - dmin)
            offset = p - slope * d for any (p,d)
        
        This function computes the slope term from the physical and digital
        values stored in the header for each channel.

        Returns: 
            A 1-D array of slope values one per signal in EDF.
        """

        pmaxes = np.array(self.physical_max)[self.channels]
        pmins = np.array(self.physical_min)[self.channels]
        dmaxes = np.array(self.digital_max)[self.channels]
        dmins = np.array(self.digital_min)[self.channels]
        return (pmaxes - pmins) / (dmaxes - dmins)

    @property
    def offsets(self):
        """Returns the offset of each channel in this Header.

        See slopes property.

        Returns: 
            A 1-D array of offset values one per signal in EDF.
        """

        pmins = np.array(self.physical_min)[self.channels]
        dmins = np.array(self.digital_min)[self.channels]
        return (pmins - self.slopes * dmins)
    
    def filter(self, indices):
        """Filters this Header keeping only data that pertains to each
        channel index in indices.

        Args:
            indices: sequence of ints
                Integer channel indices to retain in this Header.

        Returns: 
            A new header instance.
        """
        
        header = copy.deepcopy(self)
        for key, value in header.items():
            if isinstance(value, list):
                header[key] = [value[idx] for idx in indices]
        
        # update header_bytes and num_signals
        bytemap = bytemaps.edf(len(indices))
        nbytes = sum(sum(tup[0]) for tup in bytemap.values())
        header['header_bytes'] = nbytes
        header['num_signals'] = len(indices)

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
