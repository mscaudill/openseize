"""Estimators of Cross-Frequency Coupling (CFC)."""

from collections.abc import Sequence, Iterator
from types import SimpleNamespace
import multiprocessing as mp
from itertools import zip_longest
from functools import partial
import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.stats import false_discovery_control as fdr
import matplotlib.pyplot as plt

from openseize.core.producer import Producer
from openseize.filtering.bases import FIR
from openseize.core import resources
from openseize.core.mixins import ViewInstance

from openseize import producer
from openseize.core import protools
from openseize.core.mixins import ViewInstance
from openseize.filtering import fir
from openseize.filtering.special import Hilbert
from openseize.coupling.transforms import Analytic


class PhaseLock(ViewInstance):
    """An estimator of Phase-to-Power locking between two 1-D signals.

    To estimate the phase and power, this estimator uses bandlimited Hilbert
    transforms implemented as a FIR filter (Reference 1). This allows this
    estimator to scale to large data without additional memory consumption.
    However, the current implementation drops phases from each time window
    near the edge of each produced chunk of filtered values. For large
    chunksizes the error is small. This estimator is an iterative
    reimplemenation of the Canolty method (Refence 2).

    Attributes:
        hilbert:
            A Hilbert FIR filter for computing the analytic phase and amplitude.
            This filter can be initialized with: width, fs, gpass and gstop
            parameters. See filtering.special.hilbert for details.
        chunksize:
            The number of samples to hold in memory for phase and power
            computations. Larger chunksizes are advised to reduce the number of
            phases that will be dropped near each chunk edge.
        rng:
            A numpy random number generator for Monte-Carlo surrogate estimates
            of the power for statistical significance determination. This rng
            instance may be changed via the seed paramater during
            initialization.
        indices:
            The indices of the phase of interest as computed by the index method
            of this estimator. It is initialized to None.

    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Also see openseize.filtering.special"
        2. Canolty RT, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE,
           Berger MS, Barbaro NM, Knight RT. High gamma power is phase-locked to
           theta oscillations in human neocortex. Science. 2006 Sep
           15;313(5793):1626-8
    """

    def __init__(
        self,
        hilbert: Hilbert,
        chunksize: int = int(10e6),
        seed: int | None = 0,
    ) -> None:
        """Initialize this Estimator."""

        self._hilbert = hilbert
        self._chunksize = chunksize
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

    def index(
        self,
        signal: Producer | npt.NDArray,
        fpass: list[float, float],
        fstop: list[float, float],
        firfilt: FIR = fir.Kaiser,
        phase: float = 0,
        epsi: float = 0.05,
        axis: int = -1,
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
            axis:
                The sample axis of signal along which phases will be indexed.
            **kwargs:
                Keyword arguments are passed to firfilt.

        Returns:
            None but stores the phase indices to this estimator. These phase
            indices are a list of 1-D numpy arrays, one per chunk in signal.
        """

        pro = producer(signal, chunksize=self.chunksize, axis=axis)
        if pro.ndim > 1:
            "Signal to estimate phase indices must be 1D"
            raise ValueError(msg)

        # filter & analytic transform
        filt = firfilt(fpass, fstop, self.fs, **kwargs)
        x = filt(pro, chunksize=self.chunksize, axis=axis)
        analytic = Analytic(
                    x,
                    self.fs,
                    self.chunksize,
                    axis,
                    width=self.hilbert.width,
                    gpass=self.hilbert.gpass,
                    gstop=self.hilbert.gstop,
        )

        # get indices whose angle is within epsi of phase
        indices = []
        for arr in analytic.phases:
            angles = arr
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

    def _avg(self, amplitudes, indices, winpoints):
        """Returns the average power in a window centered around indices.

        This protected method is not part of this classes public API.

        Args:
            amplitudes:
                A producer or list of 1-D arrays of amplitude values.
            indices:
                A list of list of phase indices one per chunk of chunksize.
            winpoints:
                A 2-el array specifying a slice of points around each index to
                extract for averaging.

        Returns:
            A 1-D array of averaged powers of length len(range(winpoints)).
        """

        w = len(range(*winpoints))
        avg, cnt = 0, 0
        for amps, phis in zip(amplitudes, indices):
            for phi in phis:
                new_power = amps[slice(*(winpoints + phi))] ** 2
                # if power is shorter than window -> discard it
                if len(new_power) < w:
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
        axis: int,
        **kwargs,
    ) -> tuple[float, npt.NDArray, npt.NDArray]:
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
            axis:
                The axis along which the power of the  signal will be estimated.
            kwargs:
                All keyword arguments are passed to the Kaiser FIR filter.

        Returns:
            A tuple of center frequency, 1-D power signal and 1-D pvalues.
        """

        fpass = center + np.array([-bandwidth / 2, bandwidth / 2])
        fstop = fpass + np.array([-bandwidth / 2, bandwidth / 2])
        filt = fir.Kaiser(fpass, fstop, self.fs, **kwargs)
        x = filt(signal, chunksize=self.chunksize, axis=axis)
        z = protools.standardize(x, axis=axis)
        analytic = Analytic(
                    z,
                    self.fs,
                    self.chunksize,
                    axis,
                    width=self.hilbert.width,
                    gpass=self.hilbert.gpass,
                    gstop=self.hilbert.gstop,
        )

        if in_memory:
            amplitudes = [arr for arr in analytic.amplitudes]
        else:
            amplitudes = analytic.amplitudes

        # compute avg power across indices
        winpoints = np.array([-winsize // 2, winsize // 2])
        power = self._avg(amplitudes, self.indices, winpoints)
        # compute shuffle average and standard deviation
        pvalues = None
        if surrogates:
            surrogate_powers = []
            for iteration in range(surrogates):
                shuff = self.shuffle(z.shape[axis])
                surrogate_powers.append(self._avg(amplitudes, shuff, winpoints))

            mean_surrogate = np.mean(surrogate_powers, axis=0)
            std_surrogate = np.std(surrogate_powers, axis=0)
            z = (power - mean_surrogate) / (std_surrogate / np.sqrt(surrogates))
            # note this is p-value at alpha level not alpha / 2
            pvalues = 1 - stats.norm.cdf(z)

        return center, power, pvalues

    def printer(self, msg: str, verbose: bool, end="\n", flush=True) -> None:
        """Prints a msg to std out if verbose."""

        # pylint: disable-next=expression-not-assigned
        print(msg, end=end, flush=flush) if verbose else None

    def estimate(
        self,
        signal: Producer | npt.NDArray,
        centers: Sequence[float] | np.ndarray[np.float64],
        bandwidth: float = 4,
        window: float = 2,
        surrogates: int | None = 300,
        in_memory: bool = True,
        ncores: int | None = None,
        verbose: bool = True,
        axis: int = -1,
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
            axis:
                The axis along which the power of the signal will be estimated.
            kwargs:
                Keyword arguments are passed to each Kaiser filter used compute
                the amplitudes around each center frequency.

        Returns:
            A 2D array of powers and p-values of shape centers x samples where
            samples is the number of samples in window.
        """

        pro = producer(signal, chunksize=self.chunksize, axis=axis)
        if pro.ndim > 1:
            msg = "Signal must be 1-D array or Prodcuer of 1-D arrays."
            raise ValueError(msg)

        # allocate upto available cores and partial accepting on center freq.
        cores = resources.allocate(len(centers), ncores)
        func = partial(
            self._estimate,
            pro,
            bandwidth=bandwidth,
            winsize=window * self.fs,
            surrogates=surrogates,
            in_memory=in_memory,
            axis=axis,
            **kwargs,
        )

        result = {}
        # multiprocess
        if cores > 1:
            start = time.perf_counter()
            msg = f"Initializing {type(self).__name__} with {cores} cores"
            self.printer(msg, verbose)

            with mp.Pool(processes=cores) as pool:
                for i, res in enumerate(pool.imap_unordered(func, centers), 1):
                    msg = f"Frequency {i} / {len(centers)} completed"
                    self.printer(msg, verbose, end="\r")

                    center, power, pvals = res
                    pvalues = fdr(pvals) if surrogates else None
                    result[center] = [power, pvalues]

            delta = time.perf_counter() - t0
            msg = f"{type(self).__name__} estimate completed in {delta} secs"
            self.printer(msg, verbose)

        # single process
        else:
            for index, center in enumerate(centers):
                c, power, pvals = func(center)
                pvalues = fdr(pvals) if surrogates else None
                result[c] = [power, pvalues]

        # sort & stack results
        powers = np.stack([result[c][0] for c in centers])
        pvalues = np.stack([result[c][1] for c in centers])

        return powers, pvalues

    def plot(
        self,
        centers,
        powers,
        pvalues,
        window,
        alpha = 0.002,
        mpl_ax=None,
        center=True,
        **kwargs,
    ) -> None:
        """Constructs a plot of the average phase-indexed windowed power at each
        center frequency.

        Args:
            centers:
                A 1-D sequence or array of center frequencies (in Hz) at which
                the powers were be estimated.
            powers:
                A 2D array of average powers of shape centers x windows * sample
                rate.
            pvalues:
                A 2D array of p-values of shape centers x windows * sample rate.
            window:
                The time in secs the power was calculated around each indexed
                phase.
            alpha:
                The statistical level. This value will be divided by 2 for the
                alpha/2 probability of making a Type I error for p-value
                determination.
            mpl_axis:
                The matplotlib axis to plot to. If None, an axis will be
                created.
            **kwargs:
                Any valid kwarg for matplotlibs pcolormesh.

        Returns:
            None
        """

        winsize = window * self.fs
        time = np.linspace(-(winsize) // 2, (winsize) // 2, winsize)
        _, ax = plt.subplots() if not mpl_ax else mpl_ax
        z = powers - np.mean(powers, -1, keepdims=True) if center else powers
        cmap = kwargs.pop('cmap', 'jet')
        mesh = ax.pcolormesh(time, centers, z, cmap=cmap, **kwargs)
        colorbar = plt.colorbar(mesh)

        z = pvalues < alpha / 2
        ax.contour(time, centers, z, colors='white')

        plt.show()


if __name__ == "__main__":

    from openseize.file_io.edf import Reader
    from openseize.resampling.resampling import downsample
    import time
    from pathlib import Path

    base = "/media/matt/Magnus/data/rett_eeg/eegs/"
    condition = "rtt_sham"
    #name = "No_6489_right_2022-02-09_14_58_21_(2)_annotations.edf"
    name = 'No_6492_right_2022-02-08_11_06_46_annotations.edf'
    path = Path(base) / Path(condition) / Path(name)
    csize = int(10e6)
    axis = -1
    down_fs = 500
    SEED = 2
    CENTERS = np.arange(20, 230, 2)

    x = Reader(path)
    x.channels = [3]
    xpro = producer(x, chunksize=csize, axis=axis)
    dxpro = downsample(xpro, M=10, fs=5000, chunksize=csize)
    dxpro = protools.squeeze(dxpro)


    y = Reader(path)
    y.channels = [2]
    ypro = producer(y, chunksize=csize, axis=axis)
    dypro = downsample(ypro, M=10, fs=5000, chunksize=csize)
    dypro = protools.squeeze(dypro)

    estimator = PhaseLock(Hilbert(width=4, fs=down_fs), chunksize=csize,
            seed=SEED)

    t0 = time.perf_counter()
    estimator.index(dxpro, fpass=[4, 12], fstop=[2, 14], phase=0, epsi=0.05)
    print(f"Phase events in {time.perf_counter() - t0} s")

    """
    t0 = time.perf_counter()
    c, power, pvalues = estimator._estimate(dypro, center=30, bandwidth=4,
            winsize=1000, surrogates=None, in_memory=True, axis=axis)
    print(f'Powers in {time.perf_counter() - t0} s')
    """


    powers, pvalues = estimator.estimate(dypro, centers=CENTERS)
    estimator.plot(CENTERS, powers, pvalues, window=2)

    """
    center, power, pvalues, mean_surrogate, std_surrogate = estimator._estimate(
            dypro,
            center=30,
            bandwidth=4,
            winsize=1000,
            surrogates=300,
            in_memory=True
    )
    fig, ax = plt.subplots()
    ax.plot(power - np.mean(power), label='power')
    ax.plot(pvalues, label='unadj pvals')
    ax.plot(fdr(pvalues), label='adj pvals')
    ax.plot(mean_surrogate - np.mean(mean_surrogate), label='mean surr')
    ax.plot(std_surrogate, label='std surr')
    ax.legend()
    plt.show()
    """
