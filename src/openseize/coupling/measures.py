"""Estimators of Cross-Frequency Coupling (CFC)."""

from collections.abc import Sequence
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

        self.fs = fs
        self.csize = chunksize
        self.axis = axis

    def indices(
        self,
        signal: Producer | npt.NDArray,
        fpass: list[float, float],
        fstop: list[float, float],
        analytic_width: float = 4,
        angle: float = 0,
        epsi: float = 0.1,
        **kwargs
    ) -> list[npt.NDArray]:
        """Estimates the phase indices at which powers will be measured."""

        pro = producer(signal, chunksize=self.csize, axis=self.axis)
        if all(np.array(pro.shape) > 1):
            'Signal to estimate phase indices must be 1D'
            raise ValueError(msg)

        filt = fir.Kaiser(fpass, fstop, self.fs, **kwargs)
        x = filt(pro, chunksize=self.csize, axis=self.axis)
        analytic = Analytic(x, chunksize=self.csize, axis=self.axis)
        analytic.estimate(width=analytic_width, fs=self.fs)
        indices = np.squeeze(analytic.indices(analytic.phases, angle, epsi))

        return np.squeeze(indices)

    def _reindex(self, indices, chunksize):
        """ """

        rel = np.mod(indices, chunksize)
        flips = np.flatnonzero(np.diff(rel, prepend=rel[0]) <= 0)
        slices = (slice(a, b) for a, b in zip_longest(flips, flips[1:]))

        return [indices[sl] for sl in slices]

    def _shifts(self, signal, indices, size, seed):
        """ """

        pts = signal.shape[self.axis]
        rng = np.random.default_rng(seed)
        shifts = rng.integers(low=-pts, high=pts, size=size)
        yield indices

        for shift in shifts:
            yield np.mod(indices + shift, signal.shape[self.axis])

    #TODO move centers, bandwidth, analytic width, gpass, gstop to init
    # or maybe require a filter to be given to init
    def _estimate(
        self,
        signal: Producer,
        indices: npt.NDArray,
        center: float,
        bandwidth: float,
        analytic_width: float,
        winsize: float,
        n: int | None,
        seed: int | None,
        **kwargs,
        ):
        """ """

        fpass = center + np.array([-bandwidth/2, bandwidth/2])
        fstop = fpass + np.array([-bandwidth/2, bandwidth/2])
        #TODO chk gpass and gstop
        filt = fir.Kaiser(fpass, fstop, self.fs, gpass=0.1, gstop=40)
        x = filt(signal, chunksize=self.csize, axis=self.axis)
        z = protools.standardize(x, axis=self.axis)

        analytic = Analytic(z, chunksize=self.csize, axis=self.axis)
        analytic.estimate(width=analytic_width, fs=self.fs)

        surrogates = []
        for it, ixs in enumerate(self._shifts(signal, indices, size=n,
            seed=seed), 1):

            # transform indices relative to each chunk
            reindexed = self._reindex(ixs, self.csize)

            result, cnt = 0, 0
            window = (np.array([-1, 1]) * winsize / 2 * self.fs).astype(int)
            for chunk, phase_indices in zip(analytic.amplitudes, reindexed):
                x = np.squeeze(chunk)

                for idx in phase_indices:

                    y = x[slice(*(window + idx))] ** 2
                    if len(y) < winsize * self.fs:
                        continue

                    result = (cnt * result + y) / (cnt + 1)
                    cnt += 1

            surrogates.append(result)
            print(f'{it} / {n} completed')

        return center, surrogates[0], np.mean(surrogates[1:], axis=0), np.std(surrogates[1:], axis=0)





        """
        reindexed = self._reindex(indices, self.csize)

        result = 0
        cnt = 0
        span = np.array([-1, 1]) * delay * self.fs
        for amps, idxs in zip(analytic.amplitudes, reindexed):
            x = np.squeeze(amps)
            for idx in idxs:

                y = x[slice(*(span + idx))] ** 2
                if len(y) < 2 * delay * self.fs:
                    continue

                result = (cnt * result + y) / (cnt + 1)
                cnt += 1

        return center, result
        """

    def estimate(
        self,
        signal,
        indices: npt.NDArray,
        centers: Sequence[int | float],
        bandwidth: float = 4,
        analytic_width: float = 4,
        delay: float = 1,
        ncores: int | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> npt.NDArray:
        """ """

        pro = producer(signal, chunksize=self.csize, axis=self.axis)
        if all(np.array(pro.shape) > 1):
            'Signal to estimate phase indices must be 1D'
            raise ValueError(msg)

        if indices.ndim != 1:
            msg = 'Indices must be 1D'
            raise ValueError(msg)

        cores = resources.allocate(len(centers), ncores)
        result = {}

        func = partial(
                self._estimate,
                pro,
                indices,
                bandwidth=bandwidth,
                analytic_width = analytic_width,
                delay=delay,
                **kwargs,
        )

        if cores > 1:

            start = time.perf_counter()

            msg = f'Initializing {type(self).__name__} with {cores} cores'
            if verbose:
                print(msg, end='\n', flush=True)


            with mp.Pool(processes=cores) as pool:
                for idx, v  in enumerate(pool.imap_unordered(func, centers), 1):
                    msg = f'Frequency: {idx} / {len(centers)} completed'
                    if verbose:
                        print(msg, end='\r', flush=True)
                    result[v[0]] = v[1]


            dur = time.perf_counter() - t0
            msg = f'{type(self).__name__} estimate completed in {dur} secs'
            if verbose:
                print(msg)

        else:
            for index, center in enumerate(centers):
                v = func(center)
                result[v[0]] = v[1]

        return result






if __name__ == '__main__':

    from openseize.file_io.edf import Reader
    from openseize.resampling.resampling import downsample
    import time
    from pathlib import Path

    base = '/media/matt/Magnus/Qi/EEG_annotation_03272024/'
    name = 'No_6489_right_2022-02-09_14_58_21_(2)_annotations.edf'
    path = Path(base) / Path(name)

    x = Reader(path)
    x.channels = [3]
    xpro = producer(x, chunksize=10e6, axis=-1)
    dxpro = downsample(xpro, M=10, fs=5000, chunksize=int(10e6))


    y = Reader(path)
    y.channels = [2]
    ypro = producer(y, chunksize=10e6, axis=-1)
    dypro = downsample(ypro, M=10, fs=5000, chunksize=int(10e6))

    PPL = PhasePowerLocking(fs=500)

    t0 = time.perf_counter()
    indices = PPL.indices(dxpro, fpass=(4, 12), fstop=[2, 14])
    print(f'Phase events in {time.perf_counter() - t0} s')

    t0 = time.perf_counter()
    c, power, surr_power, surr_std = PPL._estimate(dypro, indices, center=165, bandwidth=4,
            analytic_width=4, winsize=2, n=30, seed=0)
    print(f'Powers in {time.perf_counter() - t0} s')

    import matplotlib.pyplot as plt
    plt.plot(surr_power)
    plt.plot(surr_power + surr_std)
    plt.plot(power)
    plt.show()

