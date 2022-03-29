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


class FIR(abc.ABC, mixins.ViewInstance):
    """ """

    def __init__(self, bands, gains, gpass, gstop, fs, **kwargs):
        """ """

        # gpass and gstop should be min and max of all bands
        #
        self.bands = bands
        self.gains = gains
        self.gpass = gpass
        self.gstop = gstop
        self.fs = fs
        self.nyq = fs / 2
        # FIXME where to put and simplify?
        starts = np.flatnonzero(np.diff(gains))
        self.width = np.min(bands[starts+1] - bands[starts])

        self.coeffs = self._build(**kwargs)

    @abc.abstractproperty
    def num_taps(self):
        """Returns the Bellanger estimate of this FIR filter's length needed
        to meet the pass and stop band attenuation criteria.

        Scipy does not have a method for estimating the number of taps
        needed for a FIR to meet pass and stop band criteria. This function
        estimates the number or taps using the Bellanger estimate. It is
        important to plot the filter using this estimate to ensure that the
        tap number is sufficient to meet the design criteria.

        Returns:
            The number of taps needed to meet the pass, stop and transition
            criteria. The returned number will always be odd to ensure no
            phase shift and integer number of samples group delay 
            (i.e. Type I FIR).

        References:
            Ballenger (2000). Digital Processing of signals: Theory and 
            Practice 3rd ed. Wiley
        """

        delta1 = 10**(-self.gpass / 20) # pass ripple
        delta2 = 10**(-self.gstop / 20) # stop attenuation
        cnt = -2/3 * np.log10(10 * delta1 * delta2) * self.fs / self.width
        # odd taps ensure integer group delay and no phase shift (Type I)
        return cnt if cnt % 2 == 1 else cnt + 1

    @abc.abstractmethod
    def _build(self, **kwargs):
        """Returns this FIR filter's coeffecients."""

    def __call__(self, data, chunksize, axis, mode='same', **kwargs):
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
            mode: str
                A numpy convolve mode; one of 'full', 'same', 'valid'.
                    Full:
                        This mode includes all points of the convolution of
                        the filter window and data. This mode does not 
                        compensate for the delay introduced by the filter.
                    Same:
                        This mode (Default) returns data of the same size
                        as the input. This mode adjust for the delay
                        introduced by the filter.
                    Valid: 
                        This mode returns values only when the filter and
                        data completely overlap. The result using this mode
                        is to shift the data (num_taps - 1) / 2 samples to
                        the left of the input data. 
            kwargs: dict
                Any valid keyword argument for the producer constructor.
                Please type help(openseize.producer) for more details.

        Returns:
            An ndarray of filtered data or a producer of ndarrays of
            filtered data. The output type will match the input data type.
        """

        pro = producer(x, chunksize, axis, **kwargs)
        # convolve to get a new producer
        result = onum.oaconvolve(pro, self.coeffs, axis, mode)
        
        # return array if input data is array
        if isinstance(x, np.ndarray):
            result = np.concantenate([arr for arr in result], axis=axis)
        
        return result



                





