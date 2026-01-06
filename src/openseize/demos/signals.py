"""

"""

import numpy as np
import numpy.typing as npt


class PAC:
    """A callable consisting of two coupled oscillations with the phase
    of one oscillation modulating the amplitude of the other oscillation.

    The call method allows this signal to be constructed for any duration,
    sampling rate and noise level.

    Attributes:
        fp:
            The frequency, in Hz, of the oscillation whose phase modulates the
            amplitude of the other oscillation.
        fa:
            The frequency, in Hz, of the oscillation whose amplitude is
            modulated by the phase of the other oscillation.
        amp_p:
            The max amplitude of the oscillation whose phase modulates the
            amplitude of the other oscillation.
        amp_a:
            The max amplitude of the oscillation whose amplitude is
            modulated by the phase of the other oscillation.
        strength:
            A value in [0,1] that encodes how strongly the amplitude of one
            oscillation is driven by the phase of the other oscillation. If 0,
            the amplitude is not driven by phase at all. If 1, the amplitude is
            completely driven by the phase.
    """

    def __init__(
        self,
        fp: float,
        fa: float,
        amp_p: float,
        amp_a: float,
        strength: float,
    ) -> None:
        """Initialize this signal."""

        self.fp = fp
        self.fa = fa
        self.amp_p = amp_p
        self.amp_a = amp_a
        self.strength = strength

    def modulated(self, time: npt.NDArray, shift: float) -> npt.NDArray:
        """Returns the 1-D amplitude modulated component of this PAC signal.

        Args:
            time:
                A 1-D vector of times over which this component will be
                computed.
            shift:
                The phase shift relative to the modulating phase at which the
                amplitude of this modulated signal is maximal.

        Returns:
            A 1-D array of the same length as time.
        """

        chi = 1 - self.strength
        phi = shift / 180 * np.pi
        modulation = ((1- chi) * np.sin(2 * np.pi * self.fp * time - shift)
                      + 1 + chi) / 2
        return self.amp_a * modulation * np.sin(2 * np.pi * self.fa * time)

    def phasic(self, time: npt.NDArray) -> npt.NDArray:
        """Returns the phase modulating component of this PAC signal.

        Args:
            time:
                A 1-D vector of times over which this component will be
                computed.

        Returns:
            A 1-D array of the same length as time.
        """

        return self.amp_p * np.sin(2 * np.pi * self.fp * time)

    def __call__(
        self,
        duration: float,
        fs: float,
        shift: float = 0,
        sigma=None,
        seed=None,
        ) -> tuple[npt.NDArray, npt.NDArray]:
        """Returns a 1-D array of times and a 1-D array of PAC signal values.

        Args:
            duration:
                The duration of the signal to create in seconds.
            fs:
                The sampling rate of the signal to create in Hz.
            shift:
                The phase relative to the modulating phase at which the
                carrier signal is maximal.
            sigma:
                The standard deviation of additive noise to this signal.
            seed:
                A random seed integer for generating random but reproducible
                signals.

        Returns:
            A 2-tuple of 1-D arrays, the times array and the PACSignal array.
        """

        rng = np.random.default_rng(seed)
        time = np.arange(0, duration, 1/fs)
        noise = rng.normal(scale=sigma, size=len(time)) if sigma else 0
        return time, self.modulated(time, shift) + self.phasic(time) + noise







if __name__ == '__main__':

    from pathlib import Path
    import matplotlib.pyplot as plt
    import scipy.signal as sps

    from openseize.file_io import edf
    from openseize.coupling.transforms import Analytic
    from openseize import producer
    from openseize.resampling.resampling import downsample
    from openseize.filtering.fir import Kaiser

    pac = PAC(fp=8, fa=40, amp_p=1.8, amp_a=1, strength=0.8)
    time, signal = pac(10, fs=500, sigma=0.25)


    """
    # openseize Analytic results
    # FIXME need to ensure that producer to analytic is at_least 2D
    analytic = Analytic(np.atleast_2d(signal), chunksize=500, axis=-1)
    analytic.estimate(width=4, fs=500)

    phase_arr = analytic.phases.to_array()
    indices = analytic.indices(analytic.phases, angle=0, epsi=0.1)

    fig, axarr = plt.subplots(2, 1)
    axarr[0].plot(time, signal)
    axarr[1].plot(time, phase_arr.flatten())
    axarr[1].scatter(indices[0]/500, np.zeros_like(indices), color='r')

    plt.show()
    """

    base = '/media/matt/Magnus/Qi/EEG_annotation_03272024/'
    name = 'No_6489_right_2022-02-09_14_58_21_(2)_annotations.edf'
    path = Path(base) / Path(name)
    csize = int(10e3)
    dfs = 500
    dtime = 800
    epsi = 0.05

    x = edf.Reader(path)
    x.channels = [3]
    xpro = producer(x, chunksize=csize, axis=-1)
    data = downsample(xpro, M=10, fs=5000, chunksize=csize).to_array()[:,
            :dtime*dfs]
    kaiser = Kaiser(fpass=[4, 12], fstop=[0, 16], fs=500, gpass=0.01, gstop=60)
    data = kaiser(data, chunksize=csize, axis=-1)
    time = np.arange(0, dtime, 1/dfs)

    # FIXME the flattening stuff is annoying so automate this

    # openseize phases and indices
    analytic = Analytic(data, chunksize=csize, axis=-1)
    analytic.estimate(width=4, fs=dfs)
    phase_arr = analytic.phases.to_array()
    indices = analytic.indices(analytic.phases, angle=0, epsi=epsi)

    #scipy phases and indices
    hilbert = sps.hilbert(data, axis=-1)
    scipy_phase = np.angle(hilbert)
    scipy_phase[scipy_phase < 0] += 2 * np.pi
    scipy_indices = np.flatnonzero(
            np.logical_and(scipy_phase > -epsi, scipy_phase < epsi)
    )

    # openseize amplitudes
    open_amplitudes = analytic.amplitudes.to_array().flatten()
    #scipy amplitudes
    scipy_amplitudes = np.abs(hilbert).flatten()

    fig, axarr = plt.subplots(2, 1, sharex=True)
    axarr[0].plot(time, data.flatten(), label='data')
    axarr[0].plot(time, open_amplitudes, label='openseize')
    axarr[0].plot(time, scipy_amplitudes, label='scipy')
    axarr[1].plot(time, phase_arr.flatten(), label='openseize')
    axarr[1].plot(time, scipy_phase.flatten(), label='scipy')
    axarr[1].scatter(indices[0]/dfs, np.zeros_like(indices), color='r')
    axarr[0].legend()
    axarr[1].legend()
    plt.show()

