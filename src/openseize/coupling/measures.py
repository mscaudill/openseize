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
        fs: float,
        chunksize: int = int(10e6),
        axis: int = -1,
    ) -> None:
        """Initialize this Estimator."""

        self._fs = fs
        self._csize = chunksize
        self._axis = axis

    @property
    def fs(self):
        """Returns the immutable sampling rate of this estimator."""

        return self._fs

    @property
    def chunksize(self):
        """Returns the immutable chunksize of this estimator."""

        return self._csize

    @property
    def axis(self):
        """Returns the immutable axis attribute of this estimator."""

        return self._axis


    # TODO DOCS
    def phase_indices(
        self,
        signal: Producer | npt.NDArray,
        fpass: list[float, float],
        fstop: list[float, float],
        analytic_width: float = 4,
        angle: float = 0,
        epsi: float = 0.05,
        **kwargs
    ) -> list[npt.NDArray]:
        """Estimates the phase indices at which powers will be measured."""

        pro = producer(signal, chunksize=self.chunksize, axis=self.axis)
        if pro.ndim > 2 or min(pro.shape) > 1:
            'Signal to estimate phase indices must be 1D'
            raise ValueError(msg)

        filt = fir.Kaiser(fpass, fstop, self.fs, **kwargs)
        x = filt(pro, chunksize=self.chunksize, axis=self.axis)
        analytic = Analytic(x, chunksize=self.chunksize, axis=self.axis)
        analytic.estimate(width=analytic_width, fs=self.fs)

        indices = []
        for arr in analytic.phases:
            phi = np.squeeze(arr)
            near = np.logical_and(phi > angle - epsi, phi < angle + epsi)
            indices.append(np.flatnonzero(near))

        return indices

    def shuffle(self, indices, rng):
        """Shuffles the phase indices within each chunk of this estimator."""

        csize = self.chunksize
        shift = rng.integers(0, csize)
        return [np.mod(arr + shift, csize) for arr in indices]

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


    #TODO move centers, bandwidth, analytic width, gpass, gstop to init
    # or maybe require a filter to be given to init
    def _estimate(
        self,
        signal: Producer,
        indices: list[npt.NDArray], #next use gen of list of arrays for shuffs
        center: float,
        bandwidth: float,
        analytic_width: float,
        winsize: float,
        shuffle_count: int | None,
        seed: int | None,
        in_memory: bool,
        **kwargs,
        ):
        """ """

        fpass = center + np.array([-bandwidth/2, bandwidth/2])
        fstop = fpass + np.array([-bandwidth/2, bandwidth/2])
        filt = fir.Kaiser(fpass, fstop, self.fs, gpass=0.01, gstop=60)
        x = filt(signal, chunksize=self.chunksize, axis=self.axis)
        z = protools.standardize(x, axis=self.axis)

        analytic = Analytic(z, chunksize=self.chunksize, axis=self.axis)
        analytic.estimate(width=analytic_width, fs=self.fs)

        amplitudes = (
                [arr for arr in analytic.amplitudes] if in_memory else
                analytic.amplitudes
        )
        power = self._avg(amplitudes, indices, winsize)
        shuffled_powers = []
        if shuffle_count:
            rng = np.random.default_rng(seed)
            for iteration in range(shuffle_count):
                shuffled = self.shuffle(indices, rng)
                shuffled_powers.append(self._avg(amplitudes, shuffled, winsize))
                #print(f'{iteration + 1} / {shuffle_count} complete')

        return center, power, shuffled_powers

    def estimate(
        self,
        signal,
        indices: list[npt.NDArray],
        centers: Sequence[int | float],
        bandwidth: float = 4,
        analytic_width: float = 4,
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
                analytic_width = analytic_width,
                winsize = window * self.fs,
                shuffle_count=shuffle_count,
                seed = seed,
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
    csize = int(10e5)
    axis = -1

    x = Reader(path)
    x.channels = [3]
    xpro = producer(x, chunksize=csize, axis=axis)
    dxpro = downsample(xpro, M=10, fs=5000, chunksize=csize)


    y = Reader(path)
    y.channels = [2]
    ypro = producer(y, chunksize=csize, axis=axis)
    dypro = downsample(ypro, M=10, fs=5000, chunksize=csize)

    PPL = PhasePowerLocking(fs=500, chunksize=csize)

    t0 = time.perf_counter()
    indices = PPL.phase_indices(dxpro, fpass=(4, 12), fstop=[2, 14], epsi=0.05)
    print(f'Phase events in {time.perf_counter() - t0} s')


    """
    t0 = time.perf_counter()
    c, power, shuffled_powers = PPL._estimate(dypro, indices, None, center=150, bandwidth=4,
            analytic_width=4, winsize=1000, shuffle_count=100, seed=0)
    print(f'Powers in {time.perf_counter() - t0} s')

    import matplotlib.pyplot as plt
    plt.plot(np.squeeze(power) - np.mean(power))
    avg_shuffle = np.mean(shuffled_powers, axis=0)
    std_shuffle = np.std(shuffled_powers, axis=0)
    plt.plot(avg_shuffle - np.mean(avg_shuffle))
    plt.plot(std_shuffle)
    plt.show()
    """

    result = PPL.estimate(dypro, indices, centers=[30, 40, 100, 200])

