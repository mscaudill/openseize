"""A collection of specialized FIR and IIR filters.

## Hilbert
A Type IV FIR implementation of a Hilbert transform; a transform which removes
the negative frequency range of a signal. When this is added to the original
signal it constitutes the complex analytic signal from which the amplitude
envelope and phase may be extracted.
"""

import numpy as np
import numpy.typing as npt
import scipy.signal as sps

from openseize.filtering.fir import Kaiser


class Hilbert(Kaiser):
    """A Type IV FIR Hilbert Transformer.

    This filter is constructed by truncating the impulse response and windowing
    with a Kaiser window to lessen the Gibbs edge phenomenon. Its implemented as
    a type IV (odd order & even tap number) high-pass filter passing from fpass
    up to Nyquist.

    Attributes:
        :see Kaiser

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> fs = 500
        >>> # Build 8Hz sine wave at 500 Hz sample rate 8 secs long
        >>> time = np.arange(2000) / fs
        >>> signal = np.sin(2 * np.pi * 8 * time)
        >>> # Build Hilbert transform (4Hz width does not impact 8Hz signal)
        >>> hilbert = Hilbert(fpass=12, fs=fs, gpass=0.1, gstop=40)
        >>> hilbert.plot()
        >>> # call hilbert to obtain the imaginary component
        >>> imag = hilbert(signal, chunksize=500, axis=-1)
        >>> # ask scipy to compute the imaginary comp. of analytic signal
        >>> analytic = sps.hilbert(signal)
        >>> scipy_imag = np.imag(analytic)
        >>> # plot openseize's imaginary vs scipy's exact answer
        >>> fig, ax = plt.subplots()
        >>> ax.plot(time, signal, label='original data')
        >>> ax.plot(time, imag, color='tab:orange',
        ...         label='openseize imag. component')
        >>> ax.plot(time, scipy_imag, color='k', linestyle='--',
        ...         label='scipy imag. component')
        >>> ax.legend()
        >>> plt.show()
        >>> # notice the single sample shift as type 4 group delay is not int

    Notes:
        FFT based methods require in-memory arrays. Also note that this type IV
        filter's group delay is a sample off from the FFT methods because type
        IV FIR filters have non-integer group delay.

     References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 9 Eqn. 9.40 "Multirate Signal Processing"
        2. https://en.wikipedia.org/wiki/Analytic_signal
        3. https://en.wikipedia.org/wiki/Hilbert_transform
    """

    def __init__(
            self,
            fpass: float,
            fs: int,
            gpass: float = 0.1,
            gstop: float = 40,
    ) -> None:
        """Initialize this Hilbert Transform Kaiser windowed FIR.

        Args:
            fpass:
                The start of the pass band of this high-pass FIR in the same
                units as fs and in the range (0, fs/2].
            fs:
                The sampling rate of the digital system.
            gpass:
                The maximum allowable ripple in the pass band in dB.
                Default of 0.01 dB is ~ 1% amplitude ripple.
            gstop:
                The minimum attenuation required in the stop band in dB.
                Default of 40 dB is a 99% amplitude attenuation.
        """

        super().__init__(fpass, fstop=0, fs=fs, gpass=gpass, gstop=gstop)

    @property
    def numtaps(self) -> int:
        """Return tap number needed to meet stricter of transition width
        & passband ripple criteria.

        Returns:
            The odd integer tap number.
        """

        ripple = max(self.pass_attenuation, self.gstop)
        ntaps, _ = sps.kaiserord(ripple, self.width / self.nyq)
        # type 4 has odd order and even filter length
        return ntaps + 1 if ntaps % 2 == 1 else ntaps

    def _build(self, **kwargs) -> npt.NDArray[np.float64]:
        """Returns a 1-D array of windowed filter coeffecients.

        Returns:
            An 1-D array of windowed FIR coeffecients of the Hilbert transform
            of numtaps length.

        Reference:
            1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 9 Eqn. 9.40 "Multirate Signal Processing"
        """

        order = self.numtaps - 1
        taps = np.linspace(-order/2, order/2, self.numtaps)
        coeffs = (1 - np.cos(taps * np.pi)) / (taps * np.pi)
        # window the truncated impulse response
        window = sps.get_window(('kaiser', *self.window_params), len(coeffs))

        return coeffs * window
