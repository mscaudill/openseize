"""

"""

import numpy as np
import numpy.typing as npt

from openseize import producer
from openseize.core.producer import Producer
from openseize.filtering.special import Hilbert

def analytic(
        data: Producer | npt.NDArray,
        fpass: float,
        fs: int,
        chunksize: int,
        axis: int = -1,
        gpass: float = 0.1,
        gstop: float = 40,
        standardize: bool = True,
        **kwargs,
) -> Producer | npt.NDArray:
    """ """

    hilbert = Hilbert(fpass, fs, gpass, gstop)
    pro = producer(data, chunksize, axis, **kwargs)



