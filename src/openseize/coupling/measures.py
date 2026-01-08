"""Estimators of Cross-Frequency Coupling (CFC)."""

from collections.abc import Sequence, Iterator
import multiprocessing as mp
from itertools import zip_longest
from functools import partial
import numpy as np
import numpy.typing as npt

from openseize.core.producer import Producer
from openseize.filtering.bases import FIR
from openseize.core import resources

from openseize import producer
from openseize.core import protools
from openseize.core.mixins import ViewInstance
from openseize.filtering import fir
from openseize.filtering.special import Hilbert
from openseize.coupling.transforms import Analytic


class PhasePowerLocking:
    """An estimator of the Phase to Power time-locking.

    This is the time phase-locking measure of Canolty et al. 2006.

    Attributes:
    """

    def __init__(
        self,
        hilbert: Hilbert,
        chunksize: int = int(10e6),
        axis: int = -1,
        seed: int | None = 0,
    ) -> None:
        """Initialize this Estimator."""

        self._hilbert = hilbert
        self._chunksize = chunksize
        self._axis = axis
        self.rng = np.random.default_rng(seed)
        self.indices = None

    @property
    def hilbert(self) -> Hilbert:
        """Returns the Hilbert filter of this estimator."""

        return self._hilbert

    @hilbert.setter
    def hilbert(self, value: Hilbert) -> None:
        """Sets the Hilbert filter of this estimator & resets phase indices."""

        self._hilbert = value
        self.indices = None

    @property
    def fs(self) -> float:
        """Returns the sampling rate of this estimator."""

        return self.hilbert.fs

    @fs.setter
    def fs(self, value: float) -> None:
        """Sets the sampling rate of this estimator & resets phase indices."""

        self._fs = value
        self.indices = None

    @property
    def chunksize(self):
        """Returns the chunksize of this estimator."""

        return self._chunksize

    @chunksize.setter
    def chunksize(self, value: int) -> None:
        """Sets the chunksize of this estimator and resets phase indices."""

        self._chunksize = value
        self.indices = None

    @property
    def axis(self):
        """Returns the sample axis of this estimator."""

        return self._axis

    @axis.setter
    def axis(self, value: int):
        """Sets the sample axis of this estimator and resests phase indices."""

        self._axis = value
        self.indices = None

    def index(
        self,
        signal: Producer | npt.NDArray,
        fpass: list[float, float],
        fstop: list[float, float],
        firfilt: FIR = fir.Kaiser,
        phase: float = 0,
        epsi: float = 0.05,
        **kwargs
    ) -> None:
        """Indexes the filtered signal's phases that are within epsi of angle.

        Args:
            signal:
                A 1-D producer or array of signal values.
            center:
                The center frequency in Hz around which phases will be index.
            bandwidth:
                The frequency bandwidth in Hz about the center frequency.
            firfilt:
                A FIR filter callable for filtering signal. Defaults to a Kaiser
                filter.
            phase:
                The phase value in degrees that is to be indexed. Defaults to 0,
                i.e. the phase trough.
            epsi:
                The tolerance in degrees about phase.
            **kwargs:
                Keyword arguments are passed to firfilt.

        Returns:
            None but stores the phase indices to this estimator. These phase
            indices are a list of 1-D numpy arrays, one per chunk in signal.
        """

        pro = producer(signal, chunksize=self.chunksize, axis=self.axis)
        if pro.ndim > 2 or min(pro.shape) > 1:
            'Signal to estimate phase indices must be 1D'
            raise ValueError(msg)

        # filter & analytic transform
        filt = firfilt(fpass, fstop, self.fs, **kwargs)
        x = filt(pro, chunksize=self.chunksize, axis=self.axis)
        analytic = Analytic(x, chunksize=self.chunksize, axis=self.axis)
        analytic.estimate(self.hilbert.width, self.fs, self.hilbert.gpass,
                self.hilbert.gstop)

        # get indices whose angle is within epsi of phase
        indices = []
        for arr in analytic.phases:
            angle = np.squeeze(arr)
            near = np.logical_and(angle > phase - epsi, angle < phase + epsi)
            indices.append(np.flatnonzero(near))

        self.indices = indices

    def shuffle(self, n_samples: int) -> list[npt.NDArray]:
        """Returns a list of 1-D arrays of shuffled indices.

        For constructing Monte-Carlo replicates indices are shifted. The maximum
        shift is the smaller of nsamples or this estimator's chunksize.

        Args:
            n_samples:
                The number of amplitude samples.

        Returns:
            A list of 1-D ndarrays of shifted indices one per chunk of the
            amplitude signal.
        """

        max_shift = min(self.chunksize, n_samples)
        shift = self.rng.integers(0, csize)
        return [np.mod(arr + shift, max_shift) for arr in self.indices]

    # TODO continue refactor from here....
    def _avg(self, amplitudes, indices, winsize):
        """ """

        window = np.array([-winsize/2, winsize/2]).astype(int)
        avg, cnt = 0, 0

        for amps, phis in zip(amplitudes, indices):
            x = np.squeeze(amps)

            # as chunksize changes this may lead to slightly different results
            # as some windows are dropped
            for phi in phis:
                new_power = x[slice(*(window + phi))] ** 2
                if len(new_power) < winsize:
                    continue

                avg = (cnt * avg + new_power) / (cnt + 1)
                cnt += 1

        return avg

    # should also take a firfilter
    def _estimate(
        self,
        signal: Producer,
        center: float,
        bandwidth: float,
        winsize: float,
        shuffle_count: int | None,
        in_memory: bool,
        **kwargs,
        ):
        """ """

        fpass = center + np.array([-bandwidth/2, bandwidth/2])
        fstop = fpass + np.array([-bandwidth/2, bandwidth/2])
        filt = fir.Kaiser(fpass, fstop, self.fs, **kwargs)
        x = filt(signal, chunksize=self.chunksize, axis=self.axis)
        z = protools.standardize(x, axis=self.axis)

        analytic = Analytic(z, chunksize=self.chunksize, axis=self.axis)
        analytic.estimate(self.hilbert.width, fs=self.fs)

        amplitudes = (
                [arr for arr in analytic.amplitudes] if in_memory else
                analytic.amplitudes
        )
        power = self._avg(amplitudes, self.indices, winsize)
        shuffled_powers = []
        if shuffle_count:
            for iteration in range(shuffle_count):
                shuffled = self.shuffle(z.shape[self.axis])
                shuffled_powers.append(self._avg(amplitudes, shuffled, winsize))
                print(f'{iteration + 1} / {shuffle_count} complete')

        return center, power, shuffled_powers

    def estimate(
        self,
        signal,
        indices: list[npt.NDArray],
        centers: Sequence[int | float],
        bandwidth: float = 4,
        window: float = 2,
        shuffle_count: int | None = 100,
        seed: int | None = 0,
        in_memory: bool = True,
        ncores: int | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> npt.NDArray:
        """ """

        pro = producer(signal, chunksize=self.chunksize, axis=self.axis)
        if all(np.array(pro.shape) > 1):
            'Signal to estimate phase indices must be 1D'
            raise ValueError(msg)

        cores = resources.allocate(len(centers), ncores)
        result = {}

        func = partial(
                self._estimate,
                pro,
                indices,
                bandwidth=bandwidth,
                winsize = window * self.fs,
                shuffle_count=shuffle_count,
                in_memory = in_memory,
                **kwargs,
        )

        if cores > 1:

            start = time.perf_counter()

            msg = f'Initializing {type(self).__name__} with {cores} cores'
            if verbose:
                print(msg, end='\n', flush=True)


            with mp.Pool(processes=cores) as pool:
                for idx, (c, power, spowers)  in enumerate(pool.imap_unordered(func, centers), 1):
                    msg = f'Frequency: {idx} / {len(centers)} completed'
                    if verbose:
                        print(msg, end='\r', flush=True)
                    result[c] = [
                            power,
                            np.mean(spowers, axis=0),
                            np.std(spowers, axis=0),
                            ]


            dur = time.perf_counter() - t0
            msg = f'{type(self).__name__} estimate completed in {dur} secs'
            if verbose:
                print(msg)

        else:
            for index, center in enumerate(centers):
                c, power, spowers = func(center)
                result[c] = [
                            power,
                            np.mean(spowers, axis=0),
                            np.std(spowers, axis=0),
                            ]


        return result



if __name__ == '__main__':

    from openseize.file_io.edf import Reader
    from openseize.resampling.resampling import downsample
    import time
    from pathlib import Path

    base = '/media/matt/Magnus/Qi/EEG_annotation_03272024/'
    name = 'No_6489_right_2022-02-09_14_58_21_(2)_annotations.edf'
    path = Path(base) / Path(name)
    csize = int(10e6)
    axis = -1
    down_fs = 500

    x = Reader(path)
    x.channels = [3]
    xpro = producer(x, chunksize=csize, axis=axis)
    dxpro = downsample(xpro, M=10, fs=5000, chunksize=csize)


    y = Reader(path)
    y.channels = [2]
    ypro = producer(y, chunksize=csize, axis=axis)
    dypro = downsample(ypro, M=10, fs=5000, chunksize=csize)

    PPL = PhasePowerLocking(Hilbert(width=4, fs=down_fs), chunksize=csize, axis=axis)

    t0 = time.perf_counter()
    PPL.index(dxpro, fpass=[4, 12], fstop=[2, 14], phase=0, epsi=0.05)
    print(f'Phase events in {time.perf_counter() - t0} s')


    t0 = time.perf_counter()
    c, power, shuffled_powers = PPL._estimate(dypro, center=150, bandwidth=4,
            winsize=1000, shuffle_count=100, in_memory=True)
    print(f'Powers in {time.perf_counter() - t0} s')

    import matplotlib.pyplot as plt
    plt.plot(np.squeeze(power) - np.mean(power))
    avg_shuffle = np.mean(shuffled_powers, axis=0)
    std_shuffle = np.std(shuffled_powers, axis=0)
    plt.plot(avg_shuffle - np.mean(avg_shuffle))
    plt.plot(std_shuffle)
    plt.show()

    #result = PPL.estimate(dypro, indices, centers=[30, 40, 100, 200])

