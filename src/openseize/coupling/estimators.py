"""Estimators of Cross-Frequency Coupling (CFC)."""

from collections.abc import Sequence, Iterator
from types import SimpleNamespace
import multiprocessing as mp
from itertools import zip_longest
from functools import partial
import numpy as np
import numpy.typing as npt
from scipy import stats

from openseize.core.producer import Producer
from openseize.filtering.bases import FIR
from openseize.core import resources

from openseize import producer
from openseize.core import protools
from openseize.core.mixins import ViewInstance
from openseize.filtering import fir
from openseize.filtering.special import Hilbert
from openseize.coupling.transforms import Analytic


class PhaseLock:
    """An estimator of the Phase to Power coupling in time

    This is the time phase-locking measure of Canolty et al. 2006.

    Attributes:
    """

    # as chunksize changes this may lead to slightly different results
    # as some windows are dropped
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
        **kwargs,
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
            "Signal to estimate phase indices must be 1D"
            raise ValueError(msg)

        # filter & analytic transform
        filt = firfilt(fpass, fstop, self.fs, **kwargs)
        x = filt(pro, chunksize=self.chunksize, axis=self.axis)
        analytic = Analytic(x, chunksize=self.chunksize, axis=self.axis)
        analytic.estimate(
            self.hilbert.width, self.fs, self.hilbert.gpass, self.hilbert.gstop
        )

        # get indices whose angle is within epsi of phase
        indices = []
        for arr in analytic.phases:
            angles = np.squeeze(arr)
            near = np.logical_and(angles > phase - epsi, angles < phase + epsi)
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

    def _avg(self, amplitudes, indices, window):
        """Returns the average power in a window centered around indices.

        This protected method is not part of this classes public API.

        Args:
            amplitudes:
                A producer or list of 1-D arrays of amplitude values.
            indices:
                A list of list of phase indices one per chunk of chunksize.
            window:
                A 2-el array specifying a slice of points around each index to
                extract for averaging.

        Returns:
            A 1-D array of averaged powers of length len(range(window)).
        """

        avg, cnt = 0, 0
        for amps, phis in zip(amplitudes, indices):
            x = np.squeeze(amps)
            for phi in phis:
                new_power = x[slice(*(window + phi))] ** 2
                # if power is shorter than window -> discard it
                if len(new_power) < len(range(*window)):
                    continue

                avg = (cnt * avg + new_power) / (cnt + 1)
                cnt += 1

        return avg

    def _estimate(
        self,
        signal: Producer,
        center: float,
        bandwidth: float,
        winsize: float,
        surrogates: int | None,
        in_memory: bool,
        **kwargs,
    ):
        """Returns average power & unadjusted p-values if shuffle count.

        This protected method is not part of this class' public API & should not
        be called externally.

        Args:
            signal:
                A producer of 1-D arrays of raw signal values.
            center:
                The center frequency at which to estimate the average power.
            bandwidth:
                The width in Hz about the center frequency.
            winsize:
                The size of the window in samples for averaging power.
            surrogates:
                The number of shuffles to construct surrogate averaged power.
            in_memory:
                A boolean indicating if the amplitudes should be held in-memory
                across all surrogate averages. This greatly speeds up the
                computation but for large data may not be feasible.
            kwargs:
                All keyword arguments are passed to the Kaiser FIR filter.
        """

        fpass = center + np.array([-bandwidth / 2, bandwidth / 2])
        fstop = fpass + np.array([-bandwidth / 2, bandwidth / 2])
        filt = fir.Kaiser(fpass, fstop, self.fs, **kwargs)
        x = filt(signal, chunksize=self.chunksize, axis=self.axis)
        z = protools.standardize(x, axis=self.axis)

        analytic = Analytic(z, chunksize=self.chunksize, axis=self.axis)
        analytic.estimate(self.hilbert.width, fs=self.fs)

        if in_memory:
            amplitudes = [arr for arr in analytic.amplitudes]
        else:
            amplitudes = analytic.amplitudes

        # compute avg power across indices
        window = np.array([-winsize // 2, winsize // 2])
        power = self._avg(amplitudes, self.indices, window)
        # compute shuffle average and standard deviation
        pvalues = None
        if surrogates:
            surrogate_powers = []
            for iteration in range(surrogates):
                shuffled = self.shuffle(z.shape[self.axis])
                surrogate_powers.append(self._avg(amplitudes, shuffled, window))

            mean_surrogate = np.mean(surrogate_powers, axis=0)
            std_surrogate = np.std(surrogate_powers, axis=0)
            pvalues = 1 - stats.norm.cdf(power, mean_surrogate, std_surrogate)

        return center, power, pvalues

    def printer(self, msg: str, verbose: bool, end="\n", flush=True) -> None:
        """Prints a msg to std out if verbose."""

        # pylint: disable-next=expression-not-assigned
        print(msg, end=end, flush=flush) if verbose else None

    # FIXME
    # add a plot method that takes powers and pvalues
    # add mixins for viewinstance
    # lint, type check etc
    # DOC at class level
    # TODO check pvalue overwrite and average overwrite here
    def estimate(
        self,
        signal: Producer | npt.NDArray,
        centers: Sequence[float] | np.ndarray[np.float64],
        bandwidth: float = 4,
        window: float = 2,
        surrogates: int | None = 300,
        adj_pvalues: bool = True,
        in_memory: bool = True,
        ncores: int | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> npt.NDArray:
        """Estimates the average signal power at each center frequency across
        all phase indices of this estimator.

        Args:
            signal:
                A 1-D signal array or Producer of 1-D signal arrays.
            centers:
                A 1-D sequence or array of center frequencies (in Hz) at which
                the powers will be estimated.
            bandwidth:
                The frequency bandwidth about each center over which the
                powers will be estimated.
            window:
                The width of the window in seconds centered around each phase
                index. Power will be estimated at all time-points of this
                window.
            surrogates:
                The number of shifted surrogates of the indices to construct
                surrogate powers for statistical significance.
            adj_pvalues:
                Boolean indicating if the p-values from the power at each center
                frequency should be adjusted for false discovery rate. This
                is carried out using reference 2.
            in_memory:
               Boolean indicating if the power of the signal should be held in
               memory for surrogates to reuse. The default of True is much
               faster but requires the entire 1-D signal be addressed to memory.
            ncores:
                The number of processing cores to ulilize for concurrently
                estimating the power across center frquencies. If None, all
                available cores will be utilized.
            verbose:
                Boolean indicating if the progress of the estimation should be
                printed to stdout.
            kwargs:
                Keyword arguments are passed to each Kaiser filter used compute
                the amplitudes around each center frequency.

        Returns:
            A 2D array of powers and p-values of shape centers x samples where
            samples is the number of samples in window.
        """

        pro = producer(signal, chunksize=self.chunksize, axis=self.axis)
        if all(np.array(pro.shape) > 1):
            msg = "Signal must be 1-D array or Prodcuer of 1-D arrays."
            raise ValueError(msg)

        cores = resources.allocate(len(centers), ncores)
        result = {}
        func = partial(
            self._estimate,
            pro,
            bandwidth=bandwidth,
            winsize=window * self.fs,
            surrogates=surrogates,
            in_memory=in_memory,
            **kwargs,
        )

        if cores > 1:

            start = time.perf_counter()
            msg = f"Initializing {type(self).__name__} with {cores} cores"
            self.printer(msg, verbose)

            with mp.Pool(processes=cores) as pool:
                for idx, (c, power, pvals) in enumerate(
                    pool.imap_unordered(func, centers), 1
                ):
                    msg = f"Frequency {idx} / {len(centers)} completed"
                    self.printer(msg, verbose, end="\r")
                    if adj_pvalues and pvals is not None:
                        pvalues = stats.false_discovery_control(pvals)
                    result[c] = [power, pvalues]

            delta = time.perf_counter() - t0
            msg = f"{type(self).__name__} estimate completed in {delta} secs"
            self.printer(msg, verbose)

        else:
            for index, center in enumerate(centers):
                c, power, pvalues = func(center)
                if adj_pvalues and pvals is not None:
                    pvalues = stats.false_discovery_control(pvals)
                result[c] = [power, pvalues]

        powers = np.stack([result[c][0] for c in centers])
        pvalues = np.stack([result[c][1] for c in centers])

        return powers, pvalues

    def plot(self, centers, powers, pvalues, window, **kwargs):
        """ """

        winsize = window * self.fs
        time = np.linspace(-(winsize) // 2, (winsize) // 2, winsize)



if __name__ == "__main__":

    from openseize.file_io.edf import Reader
    from openseize.resampling.resampling import downsample
    import time
    from pathlib import Path

    base = "/media/matt/Magnus/Qi/EEG_annotation_03272024/"
    name = "No_6489_right_2022-02-09_14_58_21_(2)_annotations.edf"
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

    estimator = PhaseLock(Hilbert(width=4, fs=down_fs), chunksize=csize, axis=axis)

    t0 = time.perf_counter()
    estimator.index(dxpro, fpass=[4, 12], fstop=[2, 14], phase=0, epsi=0.05)
    print(f"Phase events in {time.perf_counter() - t0} s")

    """
    t0 = time.perf_counter()
    c, power, pvalues = estimator._estimate(dypro, center=30, bandwidth=4,
            winsize=1000, shuffle_count=1000, in_memory=True)
    print(f'Powers in {time.perf_counter() - t0} s')
    """

    """
    import matplotlib.pyplot as plt
    corrected_p = stats.false_discovery_control(pvalues)
    plt.plot(power - np.mean(power))
    plt.plot(corrected_p)
    plt.show()
    """

    """
    import matplotlib.pyplot as plt
    plt.plot(power - np.mean(power))
    plt.plot(surrogate.avg - np.mean(surrogate.avg))
    plt.plot(surrogate.std)
    plt.show()
    """

    # I get different powers for 200 hz depending on len of centers, somehow
    # shuffling is impacting this result
    powers, pvalues = estimator.estimate(dypro, centers=[200],
            surrogates=300)
