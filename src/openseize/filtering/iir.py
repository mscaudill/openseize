"""A collection of callable Infinite Impulse Response filters.

IIR filters typically have far fewer coeffecients than similarly designed
FIR filters. However, IIRs are not linear phase and not guaranteed to be
stable. For these reasons **FIR filters are the recommended design** in
Openseize.  Nevertheless if you require a fast filter and are unconcerned
with correcting the filter's delay, IIRs may be the right choice. This
module provides the following IIRs:

    - Butterworth (Butter): IIR that is maximally flat in the pass band but
      has slow roll-off.
    - Chebyshev I (Cheby1): IIR that allows for ripple in the pass band and
      faster roll-off.
    - Chebyshev II (Cheby2): IIR that allows for ripple in the stop band and
      has fast roll-off.
    - Elliptical (Ellip): IIR that allows for ripple in the pass and stop
      band and has fast roll-off
    - Notch: A specialized IIR for rejecting a single frequency.

Examples:

    >>> # Design a lowpass Butterworth with a passband edge at 500 Hz and
    ... # transition width of 100 Hz. It should have no more than 0.5 dB
    ... # ripple in the pass band and reach 40 dB attenuation in the stop
    ... # band
    >>> butter = Butter(fpass=500, fstop=600, fs=5000, gpass=0.5, gstop=40)
    >>> # print the filter to see its attributes
    >>> print(butter)
    >>> # plot the filter to see its design
    >>> butter.plot()
    >>> # apply the filter to a producer of ndarrays without correcting the
    ... # phase delay (dephase=False)
    >>> result = butter(pro, chunksize=10000, axis=-1, dephase=False)
"""

from typing import Tuple, Union

import numpy as np
import scipy.signal as sps
from openseize.filtering.bases import IIR


class Butter(IIR):
    """A callable digital Butterworth IIR filter.

    This IIR filter meets both the pass and stop band attenuation criteria
    and is maximally flat in the pass band (no ripple). This lack of pass
    band ripple comes at the cost of slower roll-off compared with Chebyshev
    and Elliptical filters meaning a higher order will be required to meet
    a particular stop band specification.

    Attributes:
        : see IIR Base for attributes

    Examples:
        >>> # design a low pass filter with a max ripple of 1 dB and minimum
        ... # attenuation of 40 dB
        >>> butter = Butter(fpass=300, fstop=350, fs=5000, gpass=1,
        ...                 gstop=40)
        >>> butter.btype
        'lowpass'
        >>> butter = Butter(fpass=600, fstop=400, fs=1800, gpass=0.5,
        ...                 gstop=40)
        >>> butter.btype
        'highpass'
        >>> butter = Butter(fpass=[400, 1000], fstop=[200, 1200], fs=4000)
        >>> butter.btype
        'bandpass'
        >>> butter = Butter(fpass=[200, 1200], fstop=[400, 1000], fs=5000)
        >>> butter.btype
        'bandstop'
    """

    def __init__(self,
                 fpass: Union[float, Tuple[float, float]],
                 fstop: Union[float, Tuple[float, float]],
                 fs: int,
                 gpass: float = 1.0,
                 gstop: float = 40.0,
                 fmt: str = 'sos'
    ) -> None:
        """Initialize this Butterworth IIR filter.

        Args:
            fpass:
                The pass band edge frequency in the same units as fs OR
                a 2-el sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            fstop:
                The stop band edge frequency in the same units as fs OR
                a 2-el sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            fs:
                The sampling rate of the digital system.
            gpass:
                The maximum allowable ripple in the pass band in dB.
                Default of 1.0 dB is ~ 11% amplitude ripple.
            gstop:
                The minimum attenuation required in the stop band in dB.
                Default of 40 dB is a 99% amplitude attenuation.
            fmt:
                A scipy filter format specification. Must be one of:

                - 'sos': The second-order section cascade format. This
                  format is recommended as it is more stable than 'ba'.
                - 'ba': Transfer function format. This format is less stable
                  but requires fewer computations compared with 'sos'.
                - 'zpk': This format is not used and will be converted to
                  'sos'.
        """

        super().__init__(fpass, fstop, gpass, gstop, fs, fmt)

    @property
    def order(self):
        """Returns the minimum order and 3 dB point of this filter."""

        return sps.buttord(self.fpass, self.fstop, self.gpass, self.gstop,
                             fs=self.fs)


class Cheby1(IIR):
    """A callable digital Chebyshev type I IIR filter.

    This IIR filter meets both pass and stop band attenuation criteria with
    a steeper roll-off than Butterworth filters but at the expense of ripple
    in the pass band.

    Attributes:
        : see IIR Base for attributes

    Examples:
        >>> # design a low pass filter with a max ripple of 1 dB and minimum
        ... # attenuation of 40 dB
        >>> cheby1 = Cheby1(fpass=300, fstop=350, fs=5000, gpass=1,
        ...                 gstop=40)
        >>> cheby1.btype
        'lowpass'
        >>> cheby1 = Cheby1(fpass=600, fstop=400, fs=1800, gpass=0.5,
        ...                 gstop=40)
        >>> cheby1.btype
        'highpass'
        >>> cheby1 = Cheby1(fpass=[400, 1000], fstop=[200, 1200], fs=4000)
        >>> cheby1.btype
        'bandpass'
        >>> cheby1 = Cheby1(fpass=[200, 1200], fstop=[400, 1000], fs=5000)
        >>> cheby1.btype
        'bandstop'
    """

    def __init__(self,
                 fpass: Union[float, Tuple[float, float]],
                 fstop: Union[float, Tuple[float, float]],
                 fs: int,
                 gpass: float = 1.0,
                 gstop: float = 40.0,
                 fmt: str = 'sos'
    ) -> None:
        """Initialize this Chebyshev Type I IIR filter.

        Args:
            fpass:
                The pass band edge frequency in the same units as fs OR
                a 2-el sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            fstop:
                The stop band edge frequency in the same units as fs OR
                a 2-el sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            fs:
                The sampling rate of the digital system.
            gpass:
                The maximum allowable ripple in the pass band in dB.
                Default of 1.0 dB is ~ 11% amplitude ripple.
            gstop:
                The minimum attenuation required in the stop band in dB.
                Default of 40 dB is a 99% amplitude attenuation.
            fmt:
                A scipy filter format specification. Must be one of:

                - 'sos': The second-order section cascade format. This
                  format is recommended as it is more stable than 'ba'.
                - 'ba': Transfer function format. This format is less stable
                  but requires fewer computations compared with 'sos'.
                - 'zpk': This format is not used and will be converted to
                  'sos'.
        """

        super().__init__(fpass, fstop, gpass, gstop, fs, fmt)

    @property
    def order(self):
        """Returns the minimum order and 3 dB point of this filter."""

        return sps.cheb1ord(self.fpass, self.fstop, self.gpass, self.gstop,
                             fs=self.fs)


class Cheby2(IIR):
    """A callable digital Chebyshev type II IIR filter.

    This IIR filter meets both pass and stop band attenuation criteria with
    a steeper roll-off than Butterworth filters but at the expense of ripple
    in the stop band.

    Attributes:
        : see IIR Base for attributes

    Examples:
        >>> # design a low pass filter with a max ripple of 1 dB and minimum
        ... # attenuation of 40 dB
        >>> cheby2 = Cheby2(fpass=300, fstop=350, fs=5000, gpass=1,
        ...                 gstop=40)
        >>> cheby2.btype
        'lowpass'
        >>> cheby2 = Cheby2(fpass=600, fstop=400, fs=1800, gpass=0.5,
        ...                 gstop=40)
        >>> cheby2.btype
        'highpass'
        >>> cheby2 = Cheby2(fpass=[400, 1000], fstop=[200, 1200], fs=4000)
        >>> cheby2.btype
        'bandpass'
        >>> cheby2 = Cheby2(fpass=[200, 1200], fstop=[400, 1000], fs=5000)
        >>> cheby2.btype
        'bandstop'
    """

    def __init__(self,
                 fpass: Union[float, Tuple[float, float]],
                 fstop: Union[float, Tuple[float, float]],
                 fs: int,
                 gpass: float = 1.0,
                 gstop: float = 40.0,
                 fmt: str = 'sos'
    ) -> None:
        """Initialize this Chebyshev Type II IIR filter.

        Args:
            fpass:
                The pass band edge frequency in the same units as fs OR
                a 2-el sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            fstop:
                The stop band edge frequency in the same units as fs OR
                a 2-el sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            fs:
                The sampling rate of the digital system.
            gpass:
                The maximum allowable ripple in the pass band in dB.
                Default of 1.0 dB is ~ 11% amplitude ripple.
            gstop:
                The minimum attenuation required in the stop band in dB.
                Default of 40 dB is a 99% amplitude attenuation.
            fmt:
                A scipy filter format specification. Must be one of:

                - 'sos': The second-order section cascade format. This
                  format is recommended as it is more stable than 'ba'.
                - 'ba': Transfer function format. This format is less stable
                  but requires fewer computations compared with 'sos'.
                - 'zpk': This format is not used and will be converted to
                  'sos'.
        """

        super().__init__(fpass, fstop, gpass, gstop, fs, fmt)

    @property
    def order(self):
        """Returns the minimum order and 3 dB point of this filter."""

        return sps.cheb2ord(self.fpass, self.fstop, self.gpass, self.gstop,
                            fs=self.fs)


class Ellip(IIR):
    """A callable digital Elliptical IIR filter.

    This IIR filter meets both the pass and stop band attenuation criteria
    with steeper roll-off than Butterworth filters but at the expense of
    ripple in both the pass and stop bands.

    Attributes:
        : see IIR Base for attributes

    Examples:
        >>> # design a low pass filter with a max ripple of 1 dB and minimum
        ... # attenuation of 40 dB
        >>> ellip = Ellip(fpass=300, fstop=350, fs=5000, gpass=1,
        ...                 gstop=40)
        >>> ellip.btype
        'lowpass'
        >>> ellip = Ellip(fpass=600, fstop=400, fs=1800, gpass=0.5,
        ...                 gstop=40)
        >>> ellip.btype
        'highpass'
        >>> ellip = Ellip(fpass=[400, 1000], fstop=[200, 1200], fs=4000)
        >>> ellip.btype
        'bandpass'
        >>> ellip = Ellip(fpass=[200, 1200], fstop=[400, 1000], fs=5000)
        >>> ellip.btype
        'bandstop'
    """

    def __init__(self,
                 fpass: Union[float, Tuple[float, float]],
                 fstop: Union[float, Tuple[float, float]],
                 fs: int,
                 gpass: float = 1.0,
                 gstop: float = 40.0,
                 fmt: str = 'sos'
    ) -> None:
        """Initialize this Elliptical IIR filter.

        Args:
            fpass:
                The pass band edge frequency in the same units as fs OR
                a 2-el sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            fstop:
                The stop band edge frequency in the same units as fs OR
                a 2-el sequence of edge frequencies that are monotonically
                increasing and in [0, fs/2].
            fs:
                The sampling rate of the digital system.
            gpass:
                The maximum allowable ripple in the pass band in dB.
                Default of 1.0 dB is ~ 11% amplitude ripple.
            gstop:
                The minimum attenuation required in the stop band in dB.
                Default of 40 dB is a 99% amplitude attenuation.
            fmt:
                A scipy filter format specification. Must be one of:

                - 'sos': The second-order section cascade format. This
                  format is recommended as it is more stable than 'ba'.
                - 'ba': Transfer function format. This format is less stable
                  but requires fewer computations compared with 'sos'.
                - 'zpk': This format is not used and will be converted to
                  'sos'.
        """

        super().__init__(fpass, fstop, gpass, gstop, fs, fmt)

    @property
    def order(self):
        """Returns the minimum order and 3 dB point of this filter."""

        return sps.ellipord(self.fpass, self.fstop, self.gpass, self.gstop,
                             fs=self.fs)


class Notch(IIR):
    """A callable second order digital Notch IIR filter.

    This IIR achieves a -3 dB attenuation at the transition band edges
    centered on a single rejection frequency.

    Attributes:
        : see IIR Base for attributes

    Examples:
        >>> # design a Notch filter around 60 Hz with a 8 Hz transition
        >>> notch = Notch(fstop=60, width=8, fs=5000)
        >>> # print the pass and stop bands
        >>> notch.fpass
        array([56., 64.])
        >>> notch.fstop
        array([60, 60])
    """

    def __init__(self, fstop: float, width: float, fs:float) -> None:
        """Initialize this Second Order Notch IIR.

        Args:
            fstop:
                The stop frequency at which the filter reaches maximum
                attenuation in the same units as fs.
            width:
                The width of the trasition band centered on the stop
                frequency in the same units as fs.
            fs:
                The sampling rate of the digital system.
        """

        fpass = np.array([fstop - width / 2, fstop + width / 2])
        fstop = np.array([fstop, fstop])
        self.width = width
        # gpass is 3dB, gstop is determined by width
        super().__init__(fpass, fstop, gpass=3, gstop=None, fs=fs, fmt='ba')

    @property
    def order(self):
        """Returns the order (always 2) & the 3dB frequency of this IIR."""

        return len(self.coeffs[0]) - 1, self.fstop[0] - self.width / 2

    def _build(self):
        """Designs a second order notch filter that reaches -3 dB at the
        stop band edges."""

        center = self.fstop[0]
        return sps.iirnotch(center, center / self.width, fs=self.fs)
