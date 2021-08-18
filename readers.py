import numpy as np

from openseize import headers

class EDFRecord:
    """ """

    pass

class EDF:
    """

    """

    #signals will be a list of all signals in edf
    #channels will be a list of all ordinary signals in edf
    #annotations will be the annotation signal in edf (may be None)

    def __init__(self, path):
        """ """

        self.path = path
        self.header = headers.EDFHeader(path)

    @property
    def annotated(self):
        """Returns True if this is EDF contains annotations."""

        return True if 'EDF Annotations' in self.header.names else False
        
    @property
    def annotation(self):
        """Returns annotations signal index if present & None otherwise."""
        
        result = None
        if self.annotated:
            result = self.header.names.index('EDF Annotations') 
        return result

    @property
    def channels(self):
        """Returns the 'ordinary signal' indices."""

        signals = list(range(self.header.num_signals))
        if self.annotated:
            signals.pop(self.annotation)
        return signals

    @property
    def rates(self):
        """Returns the sampling rates of the channels."""

        res = self.header.samples_per_record / self.header.record_duration
        return res[ch for ch in self.channels]

    def transform(self, arr):
        """Linearly transforms the integers fetched from this EDF.

        The physical values p are linearly mapped from the digital values d:
        p = slope * d + offset, where the slope = 
        (pmax -pmin) / (dmax - dmin) & offset = p - slope * d for any (p,d)
        """
        #FIXME I may need to operate in-place

        pmaxes = np.array(self.header.physical_max)
        pmins = np.array(self.header.physical_min)
        dmaxes = np.array(self.header.digital_max)
        dmins = np.array(self.header.digital_min)
        slopes = ((pmaxes - pmins) / (dmaxes - dmins))[self.channels]
        offsets = (pmins - slopes * dmins)[self.channels]
        #apply to input arr along axis
        return arr * slopes + offsets

    #FIXME -- 
    #Will need a way to split annotation from channel signals
    #Will maybe need a way to tell if rates are equal across channels
