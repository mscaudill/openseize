"""A collection of callable Finite Impulse Response filters.

This module contains three distint types of FIR Filters.

## General Cosine window (GCW) FIRs
A filter design that allows for the specification of pass and stop bands but
*not* attenuations. The amplitude response of the filter is determined by
the window shape alone. The following table list the GCWs available:

    - Rectangular
    - Bartlett
    - Hann
    - Hamming
    - Blackman

## Kaiser Parameterized window FIR
A filter design with an additional parameter that specifies the window
shape.  These filters can be designed to meet to the strictest criteria in
the pass and stop bands. **This filter design is the recommended FIR type in
Openseize.**

## Remez (Parks-McClellan) Optimal FIRs
A filter design that minimizes the difference between the amplitude response
of a approximating Chebyshev filter from the desired amplitude response
using the Chebyshev approximation. This filter is optimal in the sense that
it yields the lowest number of coeffecients at the cost of instability. It
supports multiple pass and stop bands.

Examples:

    >>> # Design a lowpass Kaiser FIR with a passband edge at 500 Hz and
    ... # transition width of 100 Hz. It should have no more than 0.5 dB
    ... # ripple in the pass band and reach 40 dB attenuation in the stop
    ... # band
    >>> kaiser = Kaiser(fpass=500, fstop=600, fs=5000, gpass=0.5, gstop=40)
    >>> # print the filter to see its attributes
    >>> print(kaiser)
    >>> # plot the filter to see its design
    >>> kaiser.plot()
    >>> # apply the filter to a producer of ndarrays
    >>> result = kaiser(pro, chunksize=10000, axis=-1, mode='same')
"""

from typing import Sequence, Tuple, Union

import numpy as np
import scipy.signal as sps
from openseize.filtering.bases import FIR


class Kaiser(FIR):
    """A callable Type I FIR Filter using the parametric Kaiser window.

    A parameterized window allows for Kaiser filters to be designed to meet
    the stricter of user supplied pass or stop band attenuation criteria.
    Given this increased flexibility compared to general cosine windows
    (Hamming, Hann etc), **Kaiser filters are the recommended FIR filter
    design in Openseize.**

    Attributes:
        :see FIR Base for attributes

    Examples:
        >>> # design a low pass filter with a max ripple of 1 dB and minimum
        ... # attenuation of 40 dB
        >>> kaiser = Kaiser(fpass=300, fstop=350, fs=5000, gpass=1,
        ...                 gstop=40)
        >>> kaiser.btype
        'lowpass'
        >>> kaiser = Kaiser(fpass=600, fstop=400, fs=1800, gpass=0.5,
        ... gstop=40)
        >>> rect.btype
        'highpass'
        >>> rect = Rectangular(fpass=[400, 1000], fstop=[200, 1200],
        ... fs=4000)
        >>> rect.btype
        'bandpass'
        >>> rect = Rectangular(fpass=[200, 1200], fstop=[400, 1000],
        ... fs=5000)
        >>> rect.btype
        'bandstop'

    References:
        1. Ifeachor E.C. and Jervis, B.W. (2002). Digital Signal Processing:
           A Practical Approach. Prentice Hall
        2. Oppenheim, A.V. and Schafer, R.W. (2009) "Discrete-Time Signal
           Processing" 3rd Edition. Pearson.
    """

    def __init__(self,
                 fpass: Union[float, Tuple[float, float]],
                 fstop: Union[float, Tuple[float, float]],
                 fs: int,
                 gpass: float = 1.0,
                 gstop: float = 40.0
    ) -> None:
        """Initialize this Kaiser windowed FIR.

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
        """

        super().__init__(fpass, fstop, gpass, gstop, fs)

    @property
    def numtaps(self):
        """Returns the number of taps needed to meet the stricter of the
        pass and stop band criteria."""

        ripple = max(self.pass_attenuation, self.gstop)
        ntaps, _ = sps.kaiserord(ripple, self.width / self.nyq)
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps

    @property
    def window_params(self):
        """Returns the beta parameter of this filter."""

        ripple = max(self.pass_attenuation, self.gstop)
        return [sps.kaiser_beta(ripple)]


class Rectangular(FIR):
    """A callable type I FIR using a rectangular window.

    The rectangular window has the narrowest main lobe but the highest side
    lobes. Thus, this filter has very large ripples in the pass and stop
    bands. Its main use should be for perfectly periodic signals shorter
    than the window length. It is NOT a good general purpose window.

    Attributes:
        :see FIR Base for attributes

    Window Characteristics:
        - main lobe width (MLW) = 4 pi / len(taps)
        - side lobe height (SLH) = -13.3 dB
        - side lobe roll-off rate (SLRR) = -6 dB/octave
        - approximate peak error (APE) = -21 dB

    Examples:
        >>> rect = Rectangular(fpass=300, fstop=350, fs=5000)
        >>> rect.btype
        'lowpass'
        >>> rect = Rectangular(fpass=600, fstop=400, fs=1800)
        >>> rect.btype
        'highpass'
        >>> rect = Rectangular(fpass=[400, 1000], fstop=[200, 1200],
        ... fs=4000)
        >>> rect.btype
        'bandpass'
        >>> rect = Rectangular(fpass=[200, 1200], fstop=[400, 1000],
        ... fs=5000)
        >>> rect.btype
        'bandstop'
    """

    def __init__(self,
                 fpass: Union[float, Tuple[float, float]],
                 fstop: Union[float, Tuple[float, float]],
                 fs: int
    ) -> None:
        """Initialize this Rectangular windowed FIR.

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
        """

        # for plotting, provide a gpass calculated from the peak error
        peak_err = -21
        gpass = -20 * np.log10(1 - 10 ** (peak_err / 20))
        super().__init__(fpass, fstop, gpass=gpass, gstop=peak_err, fs=fs)

    @property
    def numtaps(self):
        """Return the number of taps to meet the transition width."""

        ntaps = int(4 / (self.width / self.nyq))
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps


class Bartlett(FIR):
    """A callable type I FIR using a Bartlett (triangular) window.

    A Bartlett window has an increased main lobe width but decrease side
    lobe height compared to a rectangular window. Thus, this filter has
    lower passband ripple and stronger stop band attenuation. This filter is
    NOT recommended as a general purpose filter due to larger side lobes
    compared with Hamming and Hann windowed filters.

    Attributes:
        :see FIR Base for attributes

    Window Characteristics:
        - main lobe width (MLW) = 8 pi / len(taps)
        - side lobe height (SLH) = -26.5 dB
        - side lobe roll-off rate (SLRR) = -12 dB/octave
        - approximate peak error (APE) = -25 dB

    Examples:
        >>> bartlett = Bartlett(fpass=300, fstop=350, fs=5000)
        >>> bartlett.btype
        'lowpass'
        >>> bartlett = Bartlett(fpass=600, fstop=400, fs=1800)
        >>> bartlett.btype
        'highpass'
        >>> bartlett = Bartlett(fpass=[400, 1000], fstop=[200, 1200],
        ... fs=4000)
        >>> bartlett.btype
        'bandpass'
        >>> bartlett = Bartlett(fpass=[200, 1200], fstop=[400, 1000],
        ... fs=5000)
        >>> bartlett.btype
        'bandstop'
    """

    def __init__(self,
                 fpass: Union[float, Tuple[float, float]],
                 fstop: Union[float, Tuple[float, float]],
                 fs: int
    ) -> None:
        """Initialize this Bartlett windowed FIR.

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
        """

        # for plotting, provide a gpass calculated from the peak error
        peak_err = -25
        gpass = -20 * np.log10(1 - 10 ** (peak_err / 20))
        super().__init__(fpass, fstop, gpass=gpass, gstop=peak_err, fs=fs)

    @property
    def numtaps(self):
        """Returns the integer number of taps needed to meet the transition
        width."""

        ntaps = int(8 / (self.width / self.nyq))
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps


class Hann(FIR):
    """A callable type I FIR using a Hann window.

    The Hann window has higher side lobe heights than the Hamming window but
    a faster roll-off. Thus, this filter has higher ripple than the Hamming
    but it attenuates faster. It is a good general purpose window with
    strong attenuation and fast roll-off.

    Attributes:
        :see FIR Base for attributes

    Window Characteristics:
        - main lobe width (MLW) = 8 pi / len(taps)
        - side lobe height (SLH) = -31.5 dB
        - side lobe roll-off rate (SLRR) = -18 dB/octave
        - approximate peak error (APE) = -44 dB

    Examples:
        >>> hann = Hann(fpass=300, fstop=350, fs=5000)
        >>> hann.btype
        'lowpass'
        >>> hann = Hann(fpass=600, fstop=400, fs=1800)
        >>> hann.btype
        'highpass'
        >>> hann = Hann(fpass=[400, 1000], fstop=[200, 1200], fs=4000)
        >>> hann.btype
        'bandpass'
        >>> hann = Hann(fpass=[200, 1200], fstop=[400, 1000], fs=5000)
        >>> hann.btype
        'bandstop'
    """

    def __init__(self,
                 fpass: Union[float, Tuple[float, float]],
                 fstop: Union[float, Tuple[float, float]],
                 fs: int
    ) -> None:
        """Initialize this Hann windowed FIR.

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
        """

        # for plotting, provide a gpass calculated from the peak error
        peak_err = -44
        gpass = -20 * np.log10(1 - 10 ** (peak_err / 20))
        super().__init__(fpass, fstop, gpass=gpass, gstop=peak_err, fs=fs)

    @property
    def numtaps(self):
        """Return the number of taps to meet the transition width."""

        ntaps = int(8 / (self.width / self.nyq))
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps


class Hamming(FIR):
    """A callable type I FIR using a Hamming window.

    The Hamming window has lower side lobe heights than the Rectangular or
    hamming windows. Thus, this filter has low ripple & strong attenuation
    in its pass and stop bands but its at the cost of slower roll-off. It is
    a good general purpose window.

    Attributes:
        :see FIR Base for attributes

    Window Characteristics:
        - main lobe width (MLW) = 8 pi / len(taps)
        - side lobe height (SLH) = -43.8 dB
        - side lobe roll-off rate (SLRR) = -6 dB/octave
        - approximate peak error (APE) = -53 dB

    Examples:
        >>> hamming = Hamming(fpass=300, fstop=350, fs=5000)
        >>> hamming.btype
        'lowpass'
        >>> hamming = Hamming(fpass=600, fstop=400, fs=1800)
        >>> hamming.btype
        'highpass'
        >>> hamming = hamming(fpass=[400, 1000], fstop=[200, 1200], fs=4000)
        >>> hamming.btype
        'bandpass'
        >>> hamming = hamming(fpass=[200, 1200], fstop=[400, 1000], fs=5000)
        >>> hamming.btype
        'bandstop'
    """

    def __init__(self,
                 fpass: Union[float, Tuple[float, float]],
                 fstop: Union[float, Tuple[float, float]],
                 fs: int
    ) -> None:
        """Initialize this Hamming windowed FIR.

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
        """

        # for plotting, provide a gpass calculated from the peak error
        peak_err = -53
        gpass = -20 * np.log10(1 - 10 ** (peak_err / 20))
        super().__init__(fpass, fstop, gpass=gpass, gstop=peak_err, fs=fs)

    @property
    def numtaps(self):
        """Return the number of taps to meet the transition width."""

        ntaps = int(8 / (self.width / self.nyq))
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps


class Blackman(FIR):
    """A callable type I FIR using a Blackman window.

    A Blackman window has a wide main lobe but very low side-lobes. Thus
    this filter has very low ripple and strong attenuation in the pass and
    stop bands but at the cost of wider transition bands. This filter
    typically leads to overdesigned filters (i.e. large tap numbers) to
    reduce the transition width. Its main use is for signals in which
    very attenuation is needed.

    Attributes:
        :see FIR Base for attributes

    Window Characteristics:
        - main lobe width (MLW) = 12 pi / len(taps)
        - side lobe height (SLH) = -58.2 dB
        - side lobe roll-off rate (SLRR) = -18 dB/octave
        - approximate peak error (APE) = -74 dB

    Examples:
        >>> bman = Blackman(fpass=300, fstop=350, fs=5000)
        >>> bman.btype
        'lowpass'
        >>> bman = Blackman(fpass=600, fstop=400, fs=1800)
        >>> bman.btype
        'highpass'
        >>> bman = Blackman(fpass=[400, 1000], fstop=[200, 1200], fs=4000)
        >>> bman.btype
        'bandpass'
        >>> bman = Blackman(fpass=[200, 1200], fstop=[400, 1000], fs=5000)
        >>> bman.btype
        'bandstop'
    """

    def __init__(self,
                 fpass: Union[float, Tuple[float, float]],
                 fstop: Union[float, Tuple[float, float]],
                 fs: int
    ) -> None:
        """Initialize this Blackman windowed FIR.

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
        """

        # for plotting, provide a gpass calculated from the peak error
        peak_err = -74
        gpass = -20 * np.log10(1 - 10 ** (peak_err / 20))
        super().__init__(fpass, fstop, gpass=gpass, gstop=peak_err, fs=fs)

    @property
    def numtaps(self):
        """Return the number of taps to meet the transition width."""

        ntaps = int(12 / (self.width / self.nyq))
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps


class Remez(FIR):
    """A Parks-McClellan optimal Chebyshev FIR filter.

    This FIR is designed by minimizing:

    ```math
    E(f) = W(f) |A(f) - D(f)|
    ```

    where E(f) is a the weighted difference of the actual frequency
    response A(F) from the desired frequency response D(f) for f in
    [0, nyquist] and W(f) is a set of weights.

    Attributes:
        bands (Sequence):
            A monitonically increasing sequence of pass and stop band edge
            frequencies that must include 0 and the nyquist frequencies.
        desired (Sequence):
            A sequence of desired gains in each band of bands.
        fs (int):
            The sampling rate of the digital system.
        gpass (float):
           The maximum ripple in the pass band(s) (dB).
        gstop: float
            The minimum attenuation in the stop band(s) (dB).
        kwargs:
            Any valid kwarg for scipy's signal.remez function.

            - numtaps (int):
                The number of taps used to construct this filter. This
                overrides Remez's internal numtaps property.
            - weight (Sequence):
                A sequence the same length as desired with weights to
                relatively weight each band. If no value is passed
                the weights are the inverse of the percentage loss and
                attenuation in the pass and stop bands respectively.
            - maxiter (int):
                The number of iterations to test for convergence.
            - grid_density (int):
                Resolution of the grid remez uses to minimize E(f). If
                algorithm converges but has unexpected ripple, increasing
                this value can help. The default is 16.

    Examples:
        >>> # Design a bandpass filter that passes 400 to 800 Hz
        >>> bands = [0, 300, 400, 800, 900, 2500]
        >>> desired = [0, 1, 0]
        >>> filt = Remez(bands, desired, fs=5000, gpass=.5, gstop=40)
        >>> filt.plot()

    Notes:
        The Remez algorithm may fail to converge. It is recommended that any
        filter designed with Remez be visually inspected before use.

    References:
        1. J. H. McClellan and T. W. Parks, “A unified approach to the
           design of optimum FIR linear phase digital filters”, IEEE Trans.
           Circuit Theory, vol. CT-20, pp. 697-701, 1973.
        2. Remez algorithm: https://en.wikipedia.org/wiki/Remez_algorithm

    """

    def __init__(self,
                 bands: Sequence[float],
                 desired: Sequence[float],
                 fs: int,
                 gpass: float = 1,
                 gstop: float = 40,
                 **kwargs):
        """Initialize this Remez FIR.

        Args:
            bands:
                A monotonically increasing sequence of pass & stop band edge
                frequencies than includes the 0 and nyquist frequencies.
            desired:
                A sequence containing the desired gains (1 or 0) for each
                band in bands.
            fs:
                The sampling rate of the digital system
            gpass:
                The maximum ripple in the pass band (dB). Default is 1 dB
                which is an amplitude loss of ~ 11%. If more than 1 pass
                band is supplied in bands, the same maximum loss will be
                applied to all bands.
            gstop:
                The minimum attenuation in the stop band (dB). The default
                is 40 dB which is an amplitude loss of 99%. If more than
                1 stop band is supplied in bands, the same minimum
                attenuation is applied to all stop bands.
            kwargs:
                Any valid kwarg for scipy's signal.remez function.

                - numtaps (int):
                    The number of taps used to construct this filter. This
                    overrides Remez's internal numtaps property.
                - weight (Sequence):
                    A sequence the same length as desired with weights to
                    relatively weight each band. If no value is passed
                    the weights are the inverse of the percentage loss and
                    attenuation in the pass and stop bands respectively.
                - maxiter (int):
                    The number of iterations to test for convergence.
                - grid_density (int):
                    Resolution of the grid remez uses to minimize E(f). If
                    algorithm converges but has unexpected ripple,
                    increasing this value can help. The default is 16.
        """

        self.bands = np.array(bands).reshape(-1, 2)
        self.desired = np.array(desired, dtype=bool)

        # construct fpass and fstop from bands for plotting
        fp = self.bands[self.desired].flatten()
        fpass = fp[np.logical_and(fp > 0, fp < fs / 2)]
        fst = self.bands[~self.desired].flatten()
        fstop = fst[np.logical_and(fst > 0, fst < fs / 2)]

        # transform gpass and gstop to amplitudes
        self.delta_pass = 1 - 10 ** (-gpass / 20)
        self.delta_stop = 10 ** (-gstop / 20)
        self.delta = (self.delta_pass * self.desired +
                      self.delta_stop * (1 - self.desired))

        super().__init__(fpass, fstop, gpass, gstop, fs, **kwargs)

    @property
    def btype(self):
        """Return the string band type of this filter."""

        fp, fs = self.fpass, self.fstop
        if len(fp) < 2:
            btype = "lowpass" if fp < fs else "highpass"
        elif len(fp) == 2:
            btype = "bandstop" if fp[0] < fs[0] else "bandpass"
        else:
            btype = "multiband"
        return btype

    @property
    def numtaps(self):
        """Estimates the number of taps needed to meet this filters pass and
        stop band specifications.

        This is the Bellanger estimate for the number of taps. Strictly it
        does not apply to multiband filters as it applies a single pass and
        a single stop attenuation to each band. As such the frequency
        response of the filter should be checked to ensure that the pass and
        stop band criteria are being met.

        References:
            M. Bellanger, Digital Processing of Signals: Theory and
            Practice (3rd Edition), Wiley, Hoboken, NJ, 2000.
        """

        dp, ds = self.delta_pass, self.delta_stop
        n = -2 / 3 * np.log10(10 * dp * ds) * self.fs / self.width
        ntaps = int(np.ceil(n))
        return ntaps + 1 if ntaps % 2 == 0 else ntaps

    def _build(self, **kwargs):
        """Returns the coefficients of this Remez."""

        # Get kwargs or use defaults to pass to scipy remez
        ntaps = kwargs.pop("numtaps", self.numtaps)
        weight = kwargs.pop("weight", 1 / self.delta)
        maxiter = kwargs.pop("maxiter", 25)
        grid_density = kwargs.pop("grid_density", 16)

        return sps.remez(ntaps,
                         self.bands.flatten(),
                         self.desired,
                         weight=weight,
                         maxiter=maxiter,
                         grid_density=grid_density,
                         fs=self.fs)
