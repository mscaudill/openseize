"""
Abstract Base Classes for all of Openseize's IIR and FIR filters.
"""

import abc
from functools import partial
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import scipy.signal as sps
from openseize.core import mixins
from openseize.core import numerical as nm
from openseize.core.producer import Producer, producer
from openseize.filtering.mixins import FIRViewer, IIRViewer


class IIR(abc.ABC, IIRViewer, mixins.ViewInstance):
    """Base class for infinite impulse response filters.

    Attributes:
        fpass (np.ndarray):
            1-D numpy array of start and stop edge frequencies of this
            filter's passband(s).
        fstop (np.ndarray):
            1-D numpy array of start and stop edge frequencies of this
            filter's stopband(s).
        gpass (float):
            Maximum ripple in the passband(s) in dB.
        gstop (float):
            Minimum attenuation in the stopbands in dB.
        fs (int):
            The sampling rate of the digital system.
        fmt:
            A scipy filter coeffecient format specification. Must be one
            of:

            - 'sos': The second-order section cascade format. This format is
              recommended as it is more stable than 'ba'.
            - 'ba': Transfer function format. This format is less stable but
              requires fewer computations compared with 'sos'.
            - 'zpk': This format is not used and will be converted to 'sos'.

        nyq (float):
            The nyquist rate of the digital system, fs/2.
        coeffs (np.ndarray):
            A numpy array of filter coeffecients.

    Notes:
        This IIR ABC defines the common and expected methods of all concrete
        IIR filters in the openseize.filtering.iir module. Inheritors must
        override abstract methods and properties of this base to be
        instantiable.
    """

    def __init__(self,
                 fpass: Union[float, Sequence[float]],
                 fstop: Union[float, Sequence[float]],
                 gpass: float,
                 gstop: float,
                 fs: float,
                 fmt: str,
    ) -> None:
        """Initialize this IIR Filter.

        Args:
            fpass:
                The pass band edge frequency in the same units as fs OR
                a 2-el sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            fstop:
                The stop band edge frequency in the same units as fs OR
                a 2-el sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            gpass:
                The maximum allowable ripple in the pass band in dB.
            gstop:
                The minimum attenuation required in the stop band in dB.
            fs:
                The sampling rate of the digital system.
            fmt:
                A scipy filter coeffecient format specification. Must be one
                of:

                - 'sos': The second-order section cascade format. This
                  format is recommended as it is more stable than 'ba'.
                - 'ba': Transfer function format. This format is less stable
                  but requires fewer computations compared with 'sos'.
                - 'zpk': This format is not used and will be converted to
                  'sos'.

        Raises:
            ValueError: An error if pass & stop bands lens are unequal.
        """

        self.fs = fs
        self.nyq = fs/2
        self.fpass = np.atleast_1d(fpass)
        self.fstop = np.atleast_1d(fstop)

        #validate lens of bands
        if len(self.fpass) != len(self.fstop):
            msg = 'fpass and fstop must have the same shape, got {} and {}'
            raise ValueError(msg.format(self.fpass.shape, self.fstop.shape))

        self.gpass = gpass
        self.gstop = gstop
        self.fmt = 'sos' if fmt == 'zpk' else fmt
        # on subclass init build the filter
        self.coeffs = self._build()

    @property
    def ftype(self) -> str:
        """Returns the string name of this IIR filter."""

        return type(self).__name__.lower()

    @property
    def btype(self) -> str:
        """Returns the string band type of this IIR filter."""

        fp, fs = self.fpass, self.fstop
        if len(fp) < 2:
            btype = 'lowpass' if fp < fs else 'highpass'

        else:
            btype = 'bandstop' if fp[0] < fs[0] else 'bandpass'

        return btype

    @property
    @abc.abstractmethod
    def order(self) -> Tuple[int, float]:
        """Returns the filter order and 3dB frequency for this filter."""

    def _build(self):
        """Returns an ndarray of this filter's coeffecients in 'fmt'."""

        N, Wn = self.order
        return sps.iirfilter(N, Wn, rp=self.gpass, rs=self.gstop,
                             btype=self.btype, ftype=self.ftype,
                             output=self.fmt, fs=self.fs)

    def __call__(self,
                 data: Union[Producer, np.ndarray],
                 chunksize: int,
                 axis: int = -1,
                 dephase: bool = True,
                 zi: Optional[np.ndarray] = None,
                 **kwargs) -> Union[Producer, np.ndarray]:
        """Apply this filter to an ndarray or producer of ndarrays.

        Args:
            data:
                The data to be filtered.
            chunksize:
                The number of samples to hold in memory during filtering.
            axis:
                The axis of data along which to apply the filter. If data is
                multidimensional, the filter will be independently applied
                along all slices of axis.
            dephase:
                Removes the delay introduced by this filter by running the
                filter in the forward and reverse directions of data's
                samples.
            zi:
                Initial conditions for this filter.  The shape depends on
                the fmt attribute. For 'sos' format, zi has shape nsections
                x (...,2,...) where (...,2,...) has the same shape as data
                but with 2 along axis. For more information see lfilter_zi
                and sosfilt_zi in scipy's signal module. This argument is
                ignored if dephase is True.
            kwargs:
                Keyword arguments are passed to the producer constructor.

        Returns:
            Filtered result with type matching input 'data' parameter.
        """

        pro = producer(data, chunksize, axis, **kwargs)

        if self.fmt == 'sos':
            if dephase:
                filtfunc = nm.sosfiltfilt
            else:
                filtfunc = nm.sosfilt

        if self.fmt == 'ba':
            if dephase:
                filtfunc = nm.filtfilt
            else:
                filtfunc = nm.lfilter

        # zi is ignored if filtfunc is a forward/backward filtfilt
        result = filtfunc(pro, self.coeffs, axis, zi=zi)

        # if data is an ndarray return an ndarray
        if isinstance(data, np.ndarray):
            # pylint incorrectly believes result is gen
            result = result.to_array() # pylint: disable=no-member

        return result #type: ignore


class FIR(abc.ABC, mixins.ViewInstance, FIRViewer):
    """Base class for finite impulse response filters.

    Attributes:
        fpass (np.ndarray):
            1-D numpy array of start and stop edge frequencies of this
            filter's passband(s).
        fstop (np.ndarray):
            1-D numpy array of start and stop edge frequencies of this
            filter's stopband(s).
        gpass (float):
            Maximum ripple in the passband(s) in dB.
        gstop (float):
            Minimum attenuation in the stopbands in dB.
        fs (int):
            The sampling rate of the digital system.
        nyq (float):
            The nyquist rate of the digital system, fs/2.
        width (float):
            The minimum transition width between the pass and stopbands.
        coeffs (np.ndarray):
            A 1-D numpy array of filter coeffecients.

    Notes:
        This FIR ABC defines the common and expected methods of all concrete
        FIR filters in the openseize.filtering.fir module. Inheritors must
        override abstract methods & properties of this base to be
        instantiable.
    """

    def __init__(self,
                 fpass: Union[float, Sequence[float]],
                 fstop: Union[float, Sequence[float]],
                 gpass: float,
                 gstop: float,
                 fs: float,
                 **kwargs,
    ) -> None:
        """Initialize this FIR.

        Args:
            fpass:
                The pass band edge frequency in the same units as fs OR
                a sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            fstop:
                The stop band edge frequency in the same units as fs OR
                a sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            gpass:
                The maximum loss in the passband (dB).
            gstop:
                The minimum attenuation in the stop band (dB).
            fs:
                The sampling rate of the digital system.
        """

        self.fpass = np.atleast_1d(fpass)
        self.fstop = np.atleast_1d(fstop)

        #validate lens of bands
        if len(self.fpass) != len(self.fstop):
            msg = 'fpass and fstop must have the same shape, got {} and {}'
            raise ValueError(msg.format(self.fpass.shape, self.fstop.shape))

        self.gpass = gpass
        self.gstop = gstop
        self.fs = fs
        self.nyq = fs / 2
        self.width = np.min(np.abs(self.fstop - self.fpass))
        # on subclass init build this filter
        self.coeffs = self._build(**kwargs)

    @property
    def ftype(self):
        """Returns the string name of this FIR filter."""

        return type(self).__name__.lower()

    @property
    def btype(self):
        """Returns the string band type of this filter."""

        fp, fs = self.fpass, self.fstop
        if len(fp) < 2:
            btype = 'lowpass' if fp < fs else 'highpass'

        elif len(fp) == 2:
            btype = 'bandstop' if fp[0] < fs[0] else 'bandpass'

        else:
            msg = '{} supports only lowpass, highpass, bandpass & bandstop.'
            raise ValueError(msg.format(type(self)))

        return btype

    @property
    def pass_attenuation(self):
        """Converts the max passband ripple, gpass, into a pass band
        attenuation in dB."""

        return -20 * np.log10(1 - 10 ** (-self.gpass / 20))

    @property
    def cutoff(self):
        """Returns an ndarray of the -6 dB points of each transition
        band."""

        delta = abs(self.fstop - self.fpass) / 2
        return delta + np.min(np.stack((self.fpass, self.fstop)), axis=0)

    @property
    def window_params(self):
        """Returns parameters needed to specify the window of this FIR.

        Note:
            Some of the windows used to truncate this FIRs coeffecients may
            require more than just the number of taps. For these
            parameterized windows, subclasses should override this method
            and return the additional parameters as a sequence.
        """

        return tuple()

    @property
    @abc.abstractmethod
    def numtaps(self) -> int:
        """Returns the number of taps needed to meet this FIR filters pass
        and stop band attenuation criteria within the transition width."""

    def _build(self, **kwargs):
        """Returns the ndarray of this FIR filter's coeffecients."""

        # get the window from this FIRs name and ask for add. params
        window = (self.ftype, *self.window_params)
        return sps.firwin(self.numtaps, cutoff=self.cutoff, width=None,
                          window=window, pass_zero=self.btype,
                          scale=True, fs=self.fs, **kwargs)

    def __call__(self,
                 data: Union[Producer, np.ndarray],
                 chunksize: int,
                 axis: int = -1,
                 mode: str = 'same',
                 **kwargs) -> Union[Producer, np.ndarray]:
        """Apply this filter to an ndarray or producer of ndarrays.

        Args:
            data:
                The data to be filtered.
            chunksize:
                The number of samples to hold in memory during filtering.
            axis:
                The axis of data along which to apply the filter. If data is
                multidimensional, the filter will be independently applied
                along all slices of axis.
            mode:
                A numpy convolve mode; one of 'full', 'same', 'valid'.

                - Full:
                    This mode includes all points of the convolution of
                    the filter window and data. This mode does not
                    compensate for the delay introduced by the filter.
                - Same:
                    This mode (Default) returns data of the same size
                    as the input. This mode adjust for the delay
                    introduced by the filter.
                - Valid:
                    This mode returns values only when the filter and
                    data completely overlap. The result using this mode
                    is to shift the data (num_taps - 1) / 2 samples to
                    the left of the input data.
            kwargs:
                Any valid keyword argument for the producer constructor.

        Returns:
            Filtered result with type matching input 'data' parameter.
        """

        pro = producer(data, chunksize, axis, **kwargs)
        window = self.coeffs

        # construct overlap-add generating function & get resultant shape
        genfunc = partial(nm.oaconvolve, pro, self.coeffs, axis, mode)
        shape = nm.convolved_shape(data.shape, window.shape, mode, axis)

        # build producer from generating func.
        result = producer(genfunc, chunksize, axis, shape=shape)

        # return array if input data is array
        if isinstance(data, np.ndarray):
            # pylint incorrectly believes result is a generator
            result = result.to_array() # pylint: disable=no-member


        return result
