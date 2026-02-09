"""A collection of specialized FIR and IIR filters.

## Hilbert
A Type III FIR implementation of a Hilbert transform. This filter can be used to
construct an analytic signal, a signal whose positive frequency response is the
same as the original signal but whose negative frequency response is 0.
"""

import numpy as np
import numpy.typing as npt
import scipy.signal as sps

from openseize.filtering.fir import Kaiser


class Hilbert(Kaiser):
    """A Type III FIR Hilbert Transformer.

    This filter is constructed by truncating the impulse response and windowing
    with a Kaiser window to lessen the Gibbs edge phenomenon. Its implemented as
    a type III (even order & odd tap number) band-pass filter since the group
    delay of this type is an integer value and therefore supports construction
    of the analytic signal by addition x(t) + i*Hilbert(x(t)). Although
    Hilbert Type III is implemented as a band-pass it always rolls off to the
    nyquist meaning it operates more like a high-pass filter. Therefore only the
    width of the transition band is specified.

    Attributes:
        :see Kaiser

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> fs = 500
        >>> # Build 8Hz sine wave at 500 Hz sample rate 4 secs long
        >>> time = np.arange(2000) / fs
        >>> signal = np.sin(2 * np.pi * 8 * time)
        >>> # Build Hilbert transform (4Hz width does not impact 8Hz signal)
        >>> hilbert = Hilbert(width=12, fs=fs)
        >>> hilbert.plot()
        >>> # call hilbert to obtain the imaginary component
        >>> imag = hilbert(signal, chunksize=500, axis=-1)
        >>> # ask scipy to compute the imaginary comp. of analytic signal
        >>> analytic = sps.hilbert(signal)
        >>> scipy_imag = np.imag(analytic)
        >>> # plot openseize's imaginary vs scipy's exact answer
        >>> fig, ax = plt.subplots()
        >>> _ = ax.plot(time, signal, label='original data')
        >>> _ = ax.plot(time, imag, color='tab:orange',
        ...         label='openseize imag. component')
        >>> _ = ax.plot(time, scipy_imag, color='k', linestyle='--',
        ...         label='scipy imag. component')
        >>> _ = ax.legend()
        >>> plt.show()

     References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 9 Eqn. 9.40 "Multirate Signal Processing"
        2. https://en.wikipedia.org/wiki/Analytic_signal
        3. https://en.wikipedia.org/wiki/Hilbert_transform
    """

    def __init__(
        self,
        width: float,
        fs: int,
        gpass: float = 0.01,
        gstop: float = 60,
    ) -> None:
        """Initialize this Hilbert Transform Kaiser windowed FIR.

        Args:
            width:
                The width between the stop and bass band edges in the same units
                as fs. The stricter of the width, gpass and gstop will be used
                to determine the number of taps (see scipy kaiserord).
                A reasonable value is 1/20 the nyquist.
            fs:
                The sampling rate of the digital system.
            gpass:
                The maximum allowable ripple in the pass band in dB.
                Default of 0.01 dB is ~ 0.1% amplitude ripple.
            gstop:
                The minimum attenuation required in the stop band in dB.
                Default of 60 dB is a 99.9% amplitude attenuation.
        """

        nyq = fs / 2
        fpass: tuple[float, float] = (0 + width, nyq - width)
        super().__init__(fpass, fstop=(0, nyq), fs=fs, gpass=gpass, gstop=gstop)

    @property
    def numtaps(self) -> int:
        """Return tap number needed to meet stricter of transition width
        & passband ripple criteria.

        Returns:
            The odd integer tap number.
        """

        ripple = max(self.pass_attenuation, self.gstop)
        ntaps, _ = sps.kaiserord(ripple, self.width / self.nyq)
        assert isinstance(ntaps, int)
        # type 3 has even order and odd filter length
        return ntaps + 1 if ntaps % 2 == 0 else ntaps

    def _build(self, **kwargs) -> npt.NDArray:
        """Returns a 1-D array of windowed filter coeffecients.

        Args:
            kwargs:
                All keyword arguments are ignored.

        Returns:
            An 1-D array of windowed FIR coeffecients of the Hilbert transform
            of numtaps length.

        Reference:
            1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 9 Eqn. 9.40 "Multirate Signal Processing"
        """

        order = self.numtaps - 1
        # create taps ensuring 0-tap is 1 to avoid ZeroDiv error
        taps = np.linspace(-order / 2, order / 2, self.numtaps)
        taps[order // 2] = 1
        # compute impulse response over taps
        coeffs = (1 - np.cos(taps * np.pi)) / (taps * np.pi)
        coeffs[order // 2] = 0
        # window truncated impulse response
        window = sps.get_window(("kaiser", *self.window_params), self.numtaps)
        result: npt.NDArray = coeffs * window

        return result


if __name__ == "__main__":

    import time
    from pathlib import Path

    from openseize import producer
    from openseize.file_io.edf import Reader

    base = "/media/matt/Magnus/Qi/EEG_annotation_03272024/"
    name = "No_6489_right_2022-02-09_14_58_21_(2)_annotations.edf"
    path = Path(base) / Path(name)

    reader = Reader(path)
    pro = producer(reader, chunksize=10e6, axis=-1)
    hilbert = Hilbert(width=50, fs=5000)
    hpro = hilbert(pro, chunksize=10e6, axis=-1)
    t0 = time.perf_counter()
    x = hpro.to_array()
    print(
        f"Completed convolve with {hilbert.numtaps} tap filter in "
        f"{time.perf_counter() - t0}"
    )

    arr = reader.read(0)
    t0 = time.perf_counter()
    sps.hilbert(arr, axis=-1)
    print(f"Scipy FFT method in {time.perf_counter() - t0} secs")
    """

    import matplotlib.pyplot as plt
    fs = 5000
    # Build 8Hz sine wave at 500 Hz sample rate 8 secs long
    time = np.arange(100000) / fs
    signal = np.sin(2 * np.pi * 8 * time)
    # Build Hilbert transform (4Hz width does not impact 8Hz signal)
    hilbert = Hilbert(width=8, fs=fs)
    hilbert.plot()
    # call hilbert to obtain the imaginary component
    imag = hilbert(signal, chunksize=500, axis=-1)
    # ask scipy to compute the imaginary comp. of analytic signal
    analytic = sps.hilbert(signal)
    scipy_imag = np.imag(analytic)
    # plot openseize's imaginary vs scipy's exact answer
    fig, ax = plt.subplots()
    ax.plot(time, signal, label='original data')
    ax.plot(time, imag, color='tab:orange',
            label='openseize imag. component')
    ax.plot(time, scipy_imag, color='k', linestyle='--',
            label='scipy imag. component')
    ax.legend()
    plt.show()
    """
