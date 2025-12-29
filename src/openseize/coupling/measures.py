"""Estimators of Cross-Frequency Coupling (CFC)."""

import numpy as np
import numpy.typing as npt

from openseize.core.producer import Producer
from openseize.filtering.bases import FIR

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
        filt = fir.Kaiser(fpass, fstop, self.fs, **kwargs)
        x = filt(pro, chunksize=self.csize, axis=self.axis)
        analytic = Analytic(x, chunksize=self.csize, axis=self.axis)
        analytic.estimate(width=analytic_width, fs=self.fs)

        self.indices = analytic.indices(analytic.phases, angle=0, epsi=epsi)

    def _chunked_indices(self, indices, chunksize):
        """ """



    def _power(
        self,
        signal: Producer,
        indices: npt.NDArray,
        center: float,
        bandwidth: float,
        analytic_width: float,
        epoch: float,
        **kwargs,
        ):
        """ """

        fpass = center + np.array([-bandwidth/2, bandwidth/2])
        fstop = fpass + np.array([-bandwidth/2, bandwidth/2])
        filt = fir.Kaiser(fpass, fstop, self.fs, **kwargs)
        x = filt(signal, chunksize=self.csize, axis=self.axis)
        z = protools.standardize(x, axis=self.axis)

        analytic = Analytic(z, chunksize=self.csize, axis=self.axis)
        analytic.estimate(width=analytic_width, fs=self.fs)
        result = []
        for chunk, amplitudes in analytic.amplitudes.to_array():
            index_range = range((chunk * csize, (chunk+1) * csize)





if __name__ == '__main__':

    from openseize.file_io.edf import Reader
    from openseize.resampling.resampling import downsample
    import time
    from pathlib import Path

    base = '/media/matt/Magnus/Qi/EEG_annotation_03272024/'
    name = 'No_6489_right_2022-02-09_14_58_21_(2)_annotations.edf'
    path = Path(base) / Path(name)

    x = Reader(path)
    x.channels = [0]
    pro = producer(x, chunksize=10e6, axis=-1)
    dpro = downsample(pro, M=10, fs=5000, chunksize=int(10e6))


    PPL = PhasePowerLocking(fs=500)

    t0 = time.perf_counter()
    indices = PPL.indices(dpro, fpass=(8, 12), fstop=[4, 16])
    print(f'Phase events in {time.perf_counter() - t0} s')

    t0 = time.perf_counter()
    powers = PPL._power(dpro, indices, 80, 4, analytic_width=4, epoch=1)
    print(f'Powers in {time.perf_counter() - t0} s')

