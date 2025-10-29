"""

"""

import numpy as np
import numpy.typing as npt

from openseize import producer
from openseize.core.mixins import ViewInstance
from openseize.core.producer import Producer
from openseize.core import protools
from openseize.filtering.special import Hilbert



class Analytic(ViewInstance):
    """ """

    def __init__(self,
        data: Producer | npt.NDArray,
        width: float,
        fs: int,
        gpass: float = 0.01,
        gstop: float = 60,
        chunksize: int = 100e5,
        axis: int = -1,
        **kwargs,
    ) -> Producer | npt.NDArray:
        """ """

        hilbert = Hilbert(width, fs=fs, gpass=gpass, gstop=gstop)
        x = producer(data, chunksize, axis, **kwargs)
        y = protools.multiply(hilbert(x, chunksize, axis), complex(0, 1))

        self.analytic = protools.add(x, y)

    def _envelope(self):
        """ """

        for arr in self.analytic:
            yield np.abs(arr)


    def _phase(self):
        """ """

        for arr in self.analytic:
            phi = np.angle(arr)
            phi[phi < 0] += 2 * np.pi
            yield phi

    def envelope(self):
        """ """

        chunksize = self.analytic.chunksize
        axis = self.analytic.axis
        shape = self.analytic.shape
        return producer(self._envelope, chunksize, axis, shape=shape)

    def phase(self):
        """ """

        chunksize = self.analytic.chunksize
        axis = self.analytic.axis
        shape = self.analytic.shape
        return producer(self._phase, chunksize, axis, shape=shape)




if __name__ == '__main__':


    import matplotlib.pyplot as plt
    from scipy.signal import hilbert, chirp
    from openseize.filtering.fir import Remez

    duration, fs = 10, 400  # 1 s signal with sampling frequency of 400 Hz
    t = np.arange(int(fs*duration)) / fs  # timestamps of samples
    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*8.0*t))


    # scipy
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    scipy_phase = np.angle(analytic_signal)
    scipy_phase[scipy_phase < 0] += 2*np.pi

    # openseize
    analytic = Analytic(signal, width=8, fs=fs)
    open_envelope = analytic.envelope().to_array()
    open_phase = analytic.phase().to_array()


    # remez
    """
    filt = Remez(bands=[0, 5, 10, 180, 190, 200], desired=[0, 1, 0], fs=fs,
            type='hilbert', gpass=0.1, gstop=40)
    analytic_sig = signal + complex(0, 1)*filt(signal, chunksize=1000, axis=-1)
    amp = np.abs(analytic_sig)
    """

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex='all', tight_layout=True)
    ax0.set_title("Amplitude-modulated Chirp Signal")
    ax0.set_ylabel("Amplitude")
    ax0.plot(t, signal, label='Signal')
    ax0.plot(t, amplitude_envelope, label='Scipy Envelope')
    ax0.plot(t, open_envelope, label='Openseize Envelope')
    ax0.legend()
    ax1.set(xlabel="Time in seconds", ylabel="Phase in rad")
    ax1.plot(t, scipy_phase, 'b-', label='Scipy Phase')
    ax1.plot(t, open_phase, 'r--', label='Openseize Phase')
    ax1.legend()
    plt.show()
