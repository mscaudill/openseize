import abc
import numpy as np
import scipy.signal as sps

from openseize.core import mixins
from openseize.core.producer import producer
from openseize.core import numerical as nm
from openseize.filtering.viewer import FilterViewer


class IIR(abc.ABC, mixins.ViewInstance, FilterViewer):
    """Base class for infinite impulse response filters.

    This base class defines common and expected methods used by all concrete
    IIR filters in the iir module. It is not instantiable and all concrete
    IIRs must override all abstract methods.
    """

    def __init__(self, cutoff, width, fs, btype, pass_db, stop_db, fmt):
        """Initialize this IIR filter.

        Args:
            cutoff: float or 2-el sequence
                Frequency or Frequencies at which the filter gain drops to
                -6 dB or 1/2 of the signals amplitude.
            width: float
                The width of the transition between each pass and stop band
                in this filter.
            fs: int
                The sapling rate of the signal to be filtered.
            btype: str
                A filter type that must match one of 'lowpass', 'highpass',
                'bandpass' or 'bandstop'. If specifying a band filter cutoff
                must be a 2-el sequence.
            pass_db: float
                The maximum amplitude loss in the passband. Default is -40
                dB corresponding to a max amplitude loss of 1%.
            stop_db: float
                The minimum attenuation in the stop band. Default is 40 dB
                corresponding to an amplitude attenuation of 100.
            fmt: str
                A scipy filter format str. specifying the format of this
                filters coeffecients. Must be one of 'ba', 'zpk', or 'sos'.
                Default is to use second order sections 'sos'. This is
                encouraged this leads to stable filters when the order of
                the filter is high or the transition width is narrow.
        """

        self.fs = fs
        self.nyq = fs/2
        self.cutoff = np.atleast_1d(cutoff)
        self.norm_cutoff = self.cutoff / self.nyq
        self.width = width
        self.btype = btype
        self.pass_db = pass_db
        self.stop_db = stop_db
        self.fmt = fmt
        # get the pass and stop band edge frequencies
        self._wp, self._ws = self.edges()
        # on subclass init build the filter
        self._build()
        
    @abc.abstractproperty
    def ftype(self):
        """Returns the scipy name of this IIR filter.

        Please see scipy.signal.iirfilter ftype arg for full listing of
        available scipy filters.
        """

    @abc.abstractproperty
    def order(self):
        """Returns filter order for a concrete IIR filter needed to reach
        this filters attenuation criteria."""

    def edges(self):
        """Returns the pass & stop band edge frequencies of this filter."""

        w = self.width
        widths = {'lowpass': w/2, 
                  'highpass': -w/2, 
                  'bandpass': np.array([-w, w])/2, 
                  'bandstop': np.array([w, -w])/2}
        
        return self.cutoff + widths[self.btype]

    def _build(self):
        """Designs a digital filter of a given order that meets pass and
        stop band attenuation criteria in transition bands around the cutoff
        frequencies.

        Returns: ndarray of filter coeffecients in specified scipy 'fmt'. 
        """

        return sps.iirfilter(self.order, self.cutoff, rp=self.pass_db,
                btype=self.btype, ftype=self.ftype, output=self.fmt,
                fs=self.fs) 

