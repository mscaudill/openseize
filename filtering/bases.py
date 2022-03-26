import abc
import numpy as np
import scipy.signal as sps

from openseize.core import mixins
from openseize.core.producer import producer
from openseize.core import numerical as nm
from openseize.filtering.mixins import IIRDesign


class IIR(abc.ABC, IIRDesign, mixins.ViewInstance):
    """Base class for infinite impulse response filters.

    This base class designs an IIR filter using pass and stop band edge 
    frequencies and gains. It defines the common and expected methods of all
    concrete IIR filters in the openseize.filtering.iir module. Inheritors
    in that module must overide abstract methods and properties of this base
    to be instantiable.
    """

    def __init__(self, fpass, fstop, gpass, gstop, fs, fmt):
        """Initialize this IIR filter.

        Args:
            fpass, fstop: float or 2-el sequence
                Pass and stop band edge frequencies in units of Hz.
                For example:
                - Lowpass: fpass = 1000, fstop = 1100
                - Highpass: fpass = 2000, fstop = 1800 
                - Bandpass: fpass = [400, 800], fstop = [300, 900]
                - Bandstop: fpass = [100, 200], fstop = [120, 180]
            gpass: float
                The maximum loss in the passband (dB).
            gstop: float
                The minimum attenuation in the stop band (dB).
            fs: int
                The sapling rate of the signal to be filtered.
            fmt: str
                A scipy filter format str. specifying the format of this
                filters coeffecients. Must be one of 'ba', 'zpk', or 'sos'.
                Default is to use second order sections 'sos'. This is
                encouraged this leads to stable filters when the order of
                the filter is high or the transition width is narrow. Scipy
                does not have filters for zpk format so these are converted
                to sos.
        """

        self.fs = fs
        self.nyq = fs/2
        self.fpass = np.atleast_1d(fpass)
        self.fstop = np.atleast_1d(fstop)

        #validate lens of bands
        if len(self.fpass) != len(self.fstop):
            msg = 'fpass and fstop must have the same shape, got {} and {}'
            raise ValueError(msg.format(fpass.shape, fstop.shape))
        
        self.gpass = gpass
        self.gstop = gstop
        self.fmt = 'sos' if fmt == 'zpk' else fmt
        # on subclass init build the filter
        self.coeffs = self._build()

    @property
    def btype(self):
        """Returns the band type from the pass & stop edge frequencies."""
       
        fp, fs = self.fpass, self.fstop
        if len(fp) < 2:
            btype = 'lowpass' if fp < fs else 'highpass'
        else:
            btype = 'bandstop' if fp[0] < fs[0] else 'bandpass'
        return btype
        
    @property
    def ftype(self):
        """Returns the scipy name of this IIR filter.

        Please see scipy.signal.iirfilter ftype arg for full listing of
        available scipy filters.
        """

        return type(self).__name__.lower()

    @abc.abstractproperty
    def order(self):
        """Returns the filter order & 3dB frequency for this filter.

        Scipy includes functions to compute the order (see butterord,
        cheby1ord, cheby2ord, ellipord)
        """

    def _build(self):
        """Designs a digital filter with a minimum order such that this
        filter loses no more than gpass in the passband and has a minimum
        attenaution of gstop in the stop band. 

        Returns: ndarray of filter coeffecients in specified scipy 'fmt'. 
        """

        N, Wn = self.order
        return sps.iirfilter(N, Wn, rp=self.gpass, rs=self.gstop,
                             btype=self.btype, ftype=self.ftype, 
                             output=self.fmt, fs=self.fs)

    def call(self, data, chunksize, axis, dephase=True, zi=None, **kwargs):
        """Apply this filter to an ndarray or producer of ndarrays.

        Args:
            data: ndarray or producer of ndarrays
                The data to be filtered.
            chunksize: int
                The number of samples to hold in memory during filtering.
            axis: int
                The axis of data along which to apply the filter. If data is
                multidimensional, the filter will be independently applied
                along all slices along axis.
            dephase: bool
                If True the phase delay introduced by this filter will be
                removed by applying the filter in the forwards and backwards
                direction along data axis. If False, the filtered output  
                will be delayed relative to data.
            zi: ndarray
                An array of initial conditions (i.e. steady-state responses)
                whose shape depends on the filter coeffecient format. For 
                'sos' format, zi has shape nsections x (...,2,...) where 
                (...,2,...) has the same shape as data but with 2 along 
                axis. This shape is because each section of the sos has a 
                delay of 2 along axis. For more information see lfilter_zi 
                and sosfilt_zi in scipy's signal module. This argument is 
                ignored if dephase is True. 
            kwargs: dict
                Any valid keyword argument for the producer constructor.
                Please type help(openseize.producer) for more details.

        Returns:
            An ndarray of filtered data or a producer of ndarrays of
            filtered data. The output type will match the input data type.
        """

        pro = producer(x, chunksize, axis, **kwargs)

        if self.fmt == 'sos':
            if depahse:
                filtfunc = nm.sosfiltfilt
            else:
                filtfunc = nm.sosfilt

        if self.fmt == 'ba':
            if dephase:
                filtfunc = nm.filtfilt
            else:
                filtfunc = nm.lfilter
        
        result = filtfunc(pro, self.coeffs, chunksize, axis)
        
        if isinstance(x, np.ndarray):
            # if data is an ndarray return an ndarray
            result = np.concatenate([arr for arr in result], axis=axis)

        return result





                





