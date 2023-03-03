"""Tools for estimating the spectral content of ndarray data yielded by
a producer.

This module contains the following classes and functions:

- psd:
    A function for estimating the power spectrum of a producer or
    ndarray using Welch's method. This tool assumes the data is
    **stationary** (see spectra tutorials).

- stft:
    A function for estimating the Short-time Fourier Transform of
    a producer or ndarray. This tool is appropriate for non-stationary
    data (see spectra tutorials).

Examples:
        >>> # Compute the PSD of the demo data
        >>> # import demo data and make a producer
        >>> from openseize.demos import paths
        >>> from openseize.file_io.edf import Reader
        >>> from openseize import producer
        >>> from openseize.spectra.estimators import psd
        >>> import matplotlib.pyplot as plt
        >>> fp = paths.locate('recording_001.edf')
        >>> reader = Reader(fp)
        >>> pro = producer(reader, chunksize=10e4, axis=-1)
        >>> # Compute the PSD
        >>> n, freqs, estimate = psd(pro, fs=5000, axis=-1)
        >>> # plot the channel 0 psd
        >>> plt.plot(freqs, estimate[0])

Examples:
        >>> # Compute the PSD of the demo data
        >>> # import demo data and make a producer
        >>> from openseize.demos import paths
        >>> from openseize.file_io.edf import Reader
        >>> from openseize import producer
        >>> from openseize.spectra.estimators import stft
        >>> fp = paths.locate('recording_001.edf')
        >>> reader = Reader(fp)
        >>> pro = producer(reader, chunksize=10e4, axis=-1)
        >>> # Compute the STFT of the demo data
        >>> freqs, time, estimate = stft(pro, fs=5000, axis=-1)
        >>> freqs.shape, time.shape, estimate.shape
        ((5001,), (3776,), (4, 5001, 3776))
"""

from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from openseize import producer
from openseize.core import numerical as nm
from openseize.core.producer import Producer
from openseize.core.resources import is_assignable


def psd(data: Union[npt.NDArray[np.float64], Producer],
        fs: float,
        axis: int = -1,
        resolution: float = 0.5,
        window: str = 'hann',
        overlap: float = 0.5,
        detrend: str = 'constant',
        scaling: str = 'density'
) -> Tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A power spectrum (density) estimator using Welch's method.

    Welch's method divides data into overlapping segments and averages the
    modified periodograms for each segment. This estimator can process
    ndarrays or producer of ndarrays allowing it to estimate the PSD for
    large data sets.

    Args:
        data:
            An ndarray or producer whose power spectral is to be estimated.
        fs:
            The sampling rate of the data.
        axis:
            The sample axis of the data. The estimate will be carried out
            along this axis.
        resolution:
            The frequency resolution of the estimate in Hz. The resolution
            determines the number of DFT frequencies between [0, fs) to use
            to estimate the power spectra. The number of DFT points is
            fs // resolution. The default resolution is 0.5 Hz.
        window:
            A string name for a scipy window applied to each data segment
            before computing the periodogram of that segment. For a full
            list of windows see scipy.signal.windows.
        overlap:
            A percentage in [0, 1) of the data segment overlap . If 0 this
            estimate is equivalent to Bartletts method (2).
        detrend:
            A string indicating whether to subtract the mean ('constant') or
            subtract a linear fit ('linear') from each segment prior to
            computing the estimate for a segment.
        scaling:
            Determines the normalization to apply to the estimate. If
            'spectrum' the estimate will have units V^2 and if 'density'
            V^2 / Hz.

    Returns:
        A tuple (n, frequencies, estimate) where:

         - n is the integer number of windows averaged to estimate the PSD
         - frequencies is a 1D array at which the PSD was estimated
         - estimate is a 2-D array of PSD estimates one per channel.

    Examples:
        >>> # import demo data and make a producer
        >>> from openseize.demos import paths
        >>> from openseize.file_io.edf import Reader
        >>> from openseize import producer
        >>> from openseize.spectra.estimators import psd
        >>> import matplotlib.pyplot as plt
        >>> fp = paths.locate('recording_001.edf')
        >>> reader = Reader(fp)
        >>> pro = producer(reader, chunksize=10e4, axis=-1)
        >>> # Compute the PSD
        >>> n, freqs, estimate = psd(pro, fs=5000, axis=-1)
        >>> # plot the channel 0 psd
        >>> plt.plot(freqs, estimate[0])

    References:

    - (1) P. Welch, "The use of the fast Fourier transform for the
        estimation of power spectra: A method based on time averaging
        over short, modified periodograms", IEEE Trans. Audio
        Electroacoust. vol.15, pp. 70-73, 1967

    - (2) M.S. Bartlett, "Periodogram Analysis and Continuous
        Spectra", Biometrika, vol. 37, pp. 1-16, 1950.

    - (3) B. Porat, "A Course In Digital Signal Processing" Chapters
        4 & 13. Wiley and Sons 1997.
    """

    pro = producer(data, chunksize=int(fs), axis=axis)

    # convert requested resolution to DFT pts
    nfft = int(fs / resolution)

    # build a producer of psd estimates, one per welch segment & store
    freqs, psd_pro = nm.welch(pro, fs, nfft, window, overlap, axis, detrend,
                              scaling)

    # compute the average PSD estimate
    result = 0
    for cnt, arr in enumerate(psd_pro, 1):
        result = result + 1 / cnt * (arr - result)

    # pylint misses the cnt variable here
    #pylint: disable-next=undefined-loop-variable
    return cnt, freqs, result #type: ignore


# pylint: disable-next=too-many-arguments,too-many-locals
def stft(data: Union[npt.NDArray[np.float64], Producer],
         fs: float,
         axis: int = -1,
         resolution: float = 0.5,
         window: str = 'hann',
         overlap: float = 0.5,
         detrend: str = 'constant',
         scaling: str = 'density',
         boundary: bool = True,
         padded: bool = True,
         asarray: bool = True
) -> Tuple[npt.NDArray[np.float64],
           npt.NDArray[np.float64],
           npt.NDArray[np.float64]
           ]:
    """A Short-Time Fourier Transform estimator.

    This estimator is useful for estimating changes in the frequency or
    phase content of a non-stationary signal over time.

    The STFT breaks a sequence of data into overlapping segments and
    computes a modified (windowed) Discrete Fourier Transform for each
    segment. This results in a sequence of complex values containing the
    frequency and phase content in each segment. These segments are then
    concatenated to create an estimate of the frequency content as
    a function of time X(frequency, time). This is the STFT estimate.

    Args:
        data:
            An ndarray or producer whose stft is to be estimated.
        fs:
            The sampling rate of the data.
        axis:
            The sample axis of the data. The estimate will be
            carried out along this axis.
        resolution:
            The frequency resolution of the estimate in Hz for each segment.
            The resolution determines the number of DFT frequencies between
            [0, fs) to use to estimate the power spectra. The number of DFT
            points is fs // resolution.
        window:
            A string name for a scipy window to be applied to each data
            segment before computing the modified DFT of that segment. For
            a full list of windows see scipy.signal.windows.
        overlap:
            A percentage in [0, 1) of the data segment that should overlap
            with the next data segment. If 0 this estimate is equivalent to
            Bartletts method (2).
        detrend:
            A string indicating whether to subtract the mean ('constant') or
            subtract a linear fit ('linear') from each segment prior to
            computing the estimate for a segment. Default is 'constant'.
        scaling:
            Determines the normalization to apply to the estimate. If
            'spectrum' the modulus of the estimate squared will have units
            V^2 for each segment and if density the modulus squared of the
            estimate for each segment will have units V^2/Hz.
        boundary:
            A boolean indicating if data should be padded along axis at both
            ends with nfft//2 zeros. Here nfft is the number of dft points
            in a segment (fs/resolution). If True, the endpts of the signal
            will receive the same window weight as all other points allowing
            for accurate reconstruction of the signal from the stft estimate
            through inversion (istft not implemented).
        padded:
             A boolean indicating if the signal should be extended with
             zeros so that an integer number of windows covers the signal.
             This ensures that all of the signal is used in the estimate and
             can be recovered via inversion. In contrast psd drops the
             last segment if less than nfft in size. Default is True.
        asarray:
            A boolean indicating if the estimator should attempt to return
            the result as an ndarry. If False the returned result will be
            a producer that produces segments of the STFT estimate.

    Notes:
        Scipy allows for non-zero boundary padding. Since zero extension has
        the simple interpretation of frequency interpolation of the FFT,
        openseize only allows for zero extensions of the boundaries.

    Returns:
        A tuple (freqs, time, X):

         -  freqs: A 1-D array of frequencies at the requested resolution.
         -  time: A 1-D array of times for the segments used in the estimate.
         -  X: The STFT estimate. A channels x frequencies x time ndarray or
            a producer of ndarrays each of shape channels x frequencies. The
            length of the time axis will be nfft-overlap * nfft + 1 samples
            along axis where nfft = fs // resolution.

    Examples:
        >>> # import demo data and make a producer
        >>> from openseize.demos import paths
        >>> from openseize.file_io.edf import Reader
        >>> from openseize import producer
        >>> from openseize.spectra.estimators import stft
        >>> fp = paths.locate('recording_001.edf')
        >>> reader = Reader(fp)
        >>> pro = producer(reader, chunksize=10e4, axis=-1)
        >>> freqs, time, estimate = stft(pro, fs=5000, axis=-1)
        >>> freqs.shape, time.shape, estimate.shape
        ((5001,), (3776,), (4, 5001, 3776))

    References:

    - (1) Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
      “Discrete-Time Signal Processing”, Prentice Hall, 1999.
    - (2) B. Porat, "A Course In Digital Signal Processing" Chapters
      4 & 13. Wiley and Sons 1997.
    """

    pro = producer(data, chunksize=int(fs), axis=axis)

    # convert requested resolution to DFT pts
    nfft = int(fs / resolution)

    freqs, time, result = nm.stft(pro, fs, nfft, window, overlap, axis,
                                    detrend, scaling, boundary, padded)

    # attempt to return ndarray if requested
    if asarray:

        if is_assignable(result):
            result = np.stack(list(result), axis=-1)

    return freqs, time, result
