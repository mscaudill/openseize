"""A collection of specialized FIR and IIR filter callables for transforming data.

### Hilbert
A windowed FIR implementation of the Hilbert transform. This transform
is used to construct the analytical signal a(t) from a raw signal s(t). This
analytic signal is a complex signal whose individual samples are represented by
complex vectors encoding the instantaneous amplitude and phase.
"""

import numpy as np
import scipy.signal as sps

from openseize.filtering.fir import Kaiser


class Hilbert(Kaiser):
    """A callable Type III FIR filter approximating a Hilbert transform.

    A Hilbert transform in the frequency domain imparts a phase shift of +/-
    pi/2 to every frequency component. This transform in the time domain can be
    approximated by a bandpass Type III FIR filter. Convolution of this filter
    with the original signal gives the imaginary component of the analytical
    signal; a signal whose samples in time are complex vectors encoding both
    amplitude and phase.

    Attributes:
        :see FIR Base for attributes

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> # Build 8Hz sine wave at 250 Hz sample rate 8 secs long
        >>> time = np.arange(2000) / 250
        >>> signal = np.sin(2*np.pi * 8 * time)
        >>> # Build Hilbert transform (4Hz width does not impact 8Hz signal)
        >>> hilbert = Hilbert(width=4, fs=250)
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

    Notes:
        Scipy has a function named 'hilbert' that computes the full analytic
        signal *exactly* but requires the data to be a single in-memory array.
        Openseize's implementation works iteratively but is not an exact
        solution due to the truncation of the impulse response and windowing
        approximations.

    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 9 Eqn. 9.40 "Multirate Signal Processing"
        2. https://en.wikipedia.org/wiki/Analytic_signal
        3. https://en.wikipedia.org/wiki/Hilbert_transform
    """

    def __init__(self, width: float, fs: float, gpass: float = 0.1,
                 gstop: float = 40):
        """Initialize this Hilbert by creating a Type I Kaiser filter that meets
        the width and attenuation criteria.

        Args:
            fs:
                The sampling rate of the digital system.
            width:
                The width in Hz of the transition bands at 0 Hz and fs/2 Hz.
                If the signal to be transformed is narrow-band and not near 0 Hz
                or fs/2, this tranisition width may be wide to reduce the number
                of filter coeffecients. For example if the signal to be
                transformed contains only 8-10 Hz frequencies and fs=200 then
                a tranisition width of 4 Hz will have far fewer coeffs than 1 Hz
                and have no impact on the transformation.
            gpass:
                The maximum allowable ripple in the pass band in dB. Default is
                0.1 dB is ~ 1% ripple.
            gstop:
                The minimum attenuation required in the stop band in dB. Default
                of 40 dB is ~ 99% attenuation.
        """

        fpass = [width, fs / 2 - width]
        fstop = [0, fs / 2]
        super().__init__(fpass, fstop, fs, gpass=gpass, gstop=gstop)

    def _build(self):
        """Override Kaiser's build method to create a Type III FIR estimate of
        the Hilbert transform.

        The FIR estimate of the Hilbert transform is:

                1 - cos(m * pi) / m * pi
            h = ------------------------   if m != 0
                         m * pi

            h = 0                           if m = 0

            where m = n - order / 2

        This function computes and windows this response with a Kaiser window
        meeting the initializers pass and stop criteria.

        Returns:
            An 1-D array of windowed FIR coeffecients of the Hilbert transform
            with length numtaps computed by Kaiser's numptaps property.

        Reference:
            1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 9 Eqn. 9.40 "Multirate Signal Processing"
        """

        # order is even since Kaiser is Type I
        order = self.numtaps - 1
        m = np.linspace(-order/2, order/2, self.numtaps)
        # response at m = 0 will be zero but avoid ZeroDivision in h
        m[order//2] = 1
        h = (1 - np.cos(m * np.pi)) / (m * np.pi)
        # Type III is antisymmetric with a freq. response of 0 at 0 Hz
        h[order//2] = 0

        # window the truncated impulse response
        window = sps.get_window(('kaiser', *self.window_params), len(h))

        return h * window
