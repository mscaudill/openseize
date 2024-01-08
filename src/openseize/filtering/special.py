"""A collection of specialized FIR and IIR filter callables for transforming data.

### Hilbert
A windowed FIR implementation of the Hilbert transform. This transform
is used to construct the analytical signal a(t) from a raw signal s(t). This
analytic signal contains no negative frequency components and contains both
amplitude and phase. For details see:
https://en.wikipedia.org/wiki/Analytic_signal
"""

import numpy as np
import scipy.signal as sps

from openseize.filtering.bases import FIR
from openseize.filtering.fir import Kaiser


class Hilbert(Kaiser):
    """A callable Type III FIR filter approximating a Hilbert transform.

    A Hilbert transform in the frequency domain imparts a phase shift of +/-
    pi/2 to every frequency component. This transform in the time domain can be
    approximated by a bandpass Type III FIR filter. Convolution of this filter
    with the original signal gives the imaginary component of the analytical
    signal; a signal whose negative frequency components have been removed. The
    analytical signal representation of a narrow band signal allows for
    extraction of amplitude and phase.

    Attributes:
        :see FIR Base for attributes

    Examples:
        >>> # Get demo data and build a reader then a producer
        >>> from openseize.demos import paths
        >>> filepath = paths.locate('recording_001.edf')
        >>> from openseize.file_io.edf import Reader
        >>> reader = Reader(filepath)
        >>> pro = producer(reader, chunksize=100e3, axis=-1)
        >>> # downsample the data from 5 kHz to 250 Hz
        >>> downpro = downsample(pro, M=20, fs=5000, chunksize=100e3)
        >>> # narrow band filter the downsampled between 6-8Hz
        >>> kaiser = Kaiser(fpass=[6, 8], fstop=[5, 9], fs=250, gpass=0.1, gstop=40)
        >>> x = kaiser(downpro, chunksize=20e6
        >>> # design a Hilbert transform
        >>> hilbert = Hilbert(fs=250)
        >>> 

    Notes:
        (1) Scipy's function 'hilbert' computes the analytic signal. Openseize's
        implementation computes the Hilbert transfrom, the imaginary component
        of the analytic signal. (2) This implementation works fully iteratively
        using the overlap-add convolution algorithm common to all Openseize FIR
        filters.

    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 9 Eqn. 9.40 "Multirate Signal Processing"
        2. https://en.wikipedia.org/wiki/Analytic_signal
        3. https://en.wikipedia.org/wiki/Hilbert_transform
    """

    def __init__(self, fs, fpass=None, width=1, gpass=0.1, gstop=40):
        """Initialize this Hilbert by creating a Type I Kaiser filter that meets
        the width and attenuation criteria.

        Args:
            fs:
                The sampling rate of the digital system.
            width:
                The width in Hz of the transition bands at 0 Hz and fs/2 Hz.
                Default is a 1 Hz transition band.
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

if __name__ == '__main__':

    from openseize.file_io import edf
    import matplotlib.pyplot as plt

    from scipy import signal as sps

    path = '/media/matt/Magnus/deepseize_data/CW0259_P039_@250Hz.edf'
    reader = edf.Reader(path)
    x = reader.read(0, 100000)

    y = Kaiser(fpass=8, fstop=10, gpass=0.1, gstop=40, fs=250)(x,
            chunksize=10000, axis=-1)

    hilbert = Hilbert(fs=250, width=1, gpass=.1, gstop=40)
    hilbert.plot()


    h = hilbert(y, chunksize=10000)

    analytical = y + 1j * h
    envelope = np.abs(analytical)


    #scipy
    sp_anaytical = sps.hilbert(y)
    sp_amplitude = np.abs(sp_anaytical)
    sp_h = np.imag(sp_anaytical)

    fig, ax = plt.subplots()
    ax.plot(y[0], label='data')
    ax.plot(h[0], color='tab:orange', label='imaginary')
    ax.plot(envelope[0], color='tab:green', label='amplitude')
    #scipy results
    ax.plot(sp_amplitude[0], color='r', linestyle='--', label='scipy amplitude')
    ax.plot(sp_h[0], color='black', linestyle='--', label='scipy imaginary')
    ax.legend()
    plt.show()
