"""Transforms for computing complex-valued signals from real signals for
amplitude and phase extraction.
"""

import abc
from collections import defaultdict

import numpy as np
import numpy.typing as npt

from openseize import producer
from openseize.core import protools
from openseize.core.mixins import ViewInstance
from openseize.core.producer import Producer
from openseize.filtering.special import Hilbert


class Transform(abc.ABC, ViewInstance):
    """Abstract base declaring attributes & methods of concrete Transforms.

    All Transform implementations must provide an estimate method is expected to
    return a Producer of complex values for amplitude and phase extraction.

    Attributes:
        data:
            A producer of raw data arrays to transform under the estimate
            method.
        chunksize:
            The size of each produced array along axis.
        axis:
            The axis of data to produce arrays of chunksize length along.
    """

    def __init__(
        self,
        data: Producer | npt.NDArray,
        chunksize: int = int(10e6),
        axis: int = -1,
    ) -> None:
        """Initialize this transform with raw data."""

        self.data = producer(data, chunksize=chunksize, axis=axis)
        self.estimated: Producer | None = None
        self.chunksize = chunksize
        self.axis = axis

    @abc.abstractmethod
    def estimate(self, *args, **kwargs):
        """Stores a complex producer to this Transform's 'estimated' attr."""

    def _envelope(self):
        """A generator of amplitudes from a complex signal.

        This protected method is a helper for 'amplitudes' property.
        """

        for arr in self.estimated:
            yield np.abs(arr)

    @property
    def amplitudes(self) -> Producer:
        """Returns a producer of amplitudes from this Transformer's estimated
        complex signal.

        Returns:
            A producer of real amplitudes or None if this Transformer does not
            contain an estimated complex signal.
        """

        if not self.estimated:
            msg = "This Transformer does not contain a complex signal estimate."
            raise ValueError(msg)

        return producer(
            self._envelope,
            self.chunksize,
            self.axis,
            shape=self.estimated.shape,
        )

    def _phase(self):
        """A generator of phases from a complex signal.

        This protected method is a helper for 'phases' property
        """

        for arr in self.estimated:
            phi = np.angle(arr)
            phi[phi < 0] += 2 * np.pi
            yield phi

    @property
    def phases(self) -> Producer:
        """Returns a producer of phases from this Transformer's estimated
        complex signal.

        Returns:
            A producer of real phases or None if this Transformer does not
            contain an estimated complex signal.
        """

        if not self.estimated:
            msg = "This Transformer does not contain a complex signal estimate."
            raise ValueError(msg)

        return producer(
            self._phase,
            self.chunksize,
            self.axis,
            shape=self.estimated.shape,
        )

    def indices(self, phases: Producer, angle: float, epsi=0.05):
        """Returns the indices where phases are within epsi of angle.

        Args:
            phases:
                A producer of 2D arrays of phases.
            angle:
                An angle in [0, 2*pi] radians to return indices for.
            epsi:
                The anlge tolerance values within angle +/- epsi are considered
                to be angle.

        Returns:
            A list of numpy arrays one per 1-D slice of phases containing
            indices for that slice.
        """

        results = defaultdict(list)
        for chunk, arr in enumerate(phases):
            arr = arr.T if phases.axis == 0 else arr
            arr = np.atleast_2d(arr)
            for ch, phi in enumerate(arr):
                bools = np.logical_and(phi > angle - epsi, phi < angle + epsi)
                locs = list(np.flatnonzero(bools) + phases.chunksize * chunk)
                results[ch].extend(locs)

        return [np.array(ls) for ls in results.values()]


class Analytic(Transform):
    """The Hilbert analytic Transform.

    This Transform estimates the complex analytic signal using a Type III (even
    order, odd-tap number) FIR filter representation of the Hilbert transformer.
    Specifically the analytic signal is x(t) + i H * x(t) where x(t) is the
    original signal and H is the Hilbert transform filter. This Type III Hilbert
    has 0 gain at 0 and the nyquist and has an integer delay making it suitable
    for adding it to the original signal for the analytic signal construction.

    Examples:
        >>> from scipy.signal import chirp, hilbert
        >>> import matplotlib.pyplot as plt
        >>> # make a chirp signal modulated by a slow signal
        >>> duration, fs = 10, 400  # 1 s signal with sampling frequency of 400 Hz
        >>> t = np.arange(int(fs*duration)) / fs  # timestamps of samples
        >>> data = chirp(t, 20.0, t[-1], 100.0)
        >>> data *= (1.0 + .5 * np.sin(2.0*np.pi*8.0*t))
        >>> # use scipy to compute envelope and phases
        >>> analytic_signal = hilbert(data)
        >>> amplitude_envelope = np.abs(analytic_signal)
        >>> scipy_phase = np.angle(analytic_signal)
        >>> scipy_phase[scipy_phase < 0] += 2*np.pi
        >>> # use openseize to compute envelope and phases
        >>> analytic = Analytic(data)
        >>> analytic.estimate(width=4, fs=fs)
        >>> open_envelope = analytic.amplitudes.to_array()
        >>> open_phase = analytic.phases.to_array()
        >>> # Plot comparison betweeen scipy and openseize
        >>> fig, (ax0, ax1) = plt.subplots(nrows=2, sharex='all', tight_layout=True)
        >>> _ = ax0.set_title("Amplitude-modulated Chirp Signal")
        >>> _ = ax0.set_ylabel("Amplitude")
        >>> _ = ax0.plot(t, data, label='data')
        >>> _ = ax0.plot(t, amplitude_envelope, label='Scipy Envelope')
        >>> _ = ax0.plot(t, open_envelope, label='Openseize Envelope')
        >>> _ = ax0.legend()
        >>> _ = ax1.set(xlabel="Time in seconds", ylabel="Phase in rad")
        >>> _ = ax1.plot(t, scipy_phase, 'b-', label='Scipy Phase')
        >>> _ = ax1.plot(t, open_phase, 'r--', label='Openseize Phase')
        >>> _ = ax1.legend()
        >>> indices = analytic.indices(analytic.phases, 0, epsi=.3)
        >>> _ = ax1.scatter(np.array(indices[0])/fs, np.zeros(len(indices[0])))
        >>> plt.show()
    """

    # pylint: disable-next=arguments-differ
    def estimate(
        self,
        width: float,
        fs: int,
        *args,
        gpass: float = 0.01,
        gstop: float = 60,
        **kwargs,
    ):
        """Estimate the complex analytic signal.

        Args:
            width:
                The transition width of the Hilbert FIR filter around 0 and
                nyquist frequencies. This width should not encroach on the
                passband of the data as it will attenuate.
            fs:
                The sampling rate of the data of stored to this Transformer.
            *args:
                All other positional arguments are ignored.
            gpass:
                The maximum allowable ripple in the pass band in dB.
                Default of 0.01 dB is ~ 0.1% amplitude ripple.
            gstop:
                The minimum attenuation required in the stop band in dB.
                Default of 60 dB is a 99.9% amplitude attenuation.
            **

        Returns:
            None but stores analytic signal to 'estimated' attribute.
        """

        hilbert = Hilbert(width, fs=fs, gpass=gpass, gstop=gstop)
        real = producer(self.data, self.chunksize, self.axis)
        imag = hilbert(real, self.chunksize, self.axis)
        assert isinstance(imag, Producer)
        imag = protools.multiply(imag, complex(0, 1))

        self.estimated = protools.add(real, imag)


if __name__ == "__main__":

    import doctest

    doctest.testmod()
