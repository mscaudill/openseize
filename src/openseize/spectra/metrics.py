"""Tools for measuring quantities from a power spectrum estimate.

This module includes the following functions:

- power: A function that measures the band-power from a power spectral
   density estimate between two frequencies.

- power_norm: A function that normalizes the amplitude of a power spectral
  density by the total power between two frequencies.

- confidence interval: A function that estimates the 1-alpha confidence
  interval for a power spectrum using the Chi-squared distribution.
"""

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson
from scipy.stats import chi2

from openseize.core.arraytools import nearest1D
from openseize.core.arraytools import slice_along_axis


def power(psd: npt.NDArray[np.float64],
          freqs: npt.NDArray[np.float64],
          start: Optional[float] = None,
          stop: Optional[float]=None,
          axis: int = -1
) -> npt.NDArray[np.float64]:
    """Returns the power in an array of power spectrum densities between
    start and stop frequencies using Simpson's rule.

    Args:
        psd:
            An ndarray of power spectral densities.
        freqs:
            An 1-D array of frequencies for each psd value along axis.
        start:
            The start frequency to begin measuring power at. If not in
            freqs, the closest value in freqs will be used. If None the
            first frequency in freqs is used.
        stop:
            The stop frequency to stop measuring power at. If not in
            freqs, the closest value in freqs will be used. If None the
            last frequency in freqs is used.
        axis:
            The frequency axis of the psd ndarray.

    Returns:
        A 1-D array of powers measured over axis.

    Examples:
        >>> # Compute the PSD of the demo data
        >>> # import demo data and make a producer
        >>> from openseize.demos import paths
        >>> from openseize.file_io.edf import Reader
        >>> from openseize import producer
        >>> from openseize.spectra.estimators import psd
        >>> from openseize.spectra.metrics import power
        >>> fp = paths.locate('recording_001.edf')
        >>> reader = Reader(fp)
        >>> pro = producer(reader, chunksize=10e4, axis=-1)
        >>> # Compute the PSD
        >>> n, freqs, estimate = psd(pro, fs=5000, axis=-1)
        >>> # compute the power in the 0-40Hz for each channel
        >>> result = power(estimate, freqs, start=0, stop=40)

    Note:
        This metric should only be used on data that is density scaled.
    """

    if start is None:
        start = freqs[0]
    if stop is None:
        stop = freqs[-1]

    # compute start and stop frequency indices
    a, b = nearest1D(freqs, start), nearest1D(freqs, stop)

    # slice between freq indices inclusively & integrate
    arr = slice_along_axis(psd, start=a, stop=b+1, axis=axis)
    result = simpson(arr, dx=freqs[1]-freqs[0], axis=axis)

    return result #type: ignore


def power_norm(estimate: npt.NDArray[np.float64],
               freqs: npt.NDArray[np.float64],
               start: Optional[int] = None,
               stop: Optional[int] = None,
               axis: int = -1
) -> npt.NDArray[np.float64]:
    """Normalizes power spectral densities by the total power between
    start and stop frequencies.

    Args:
        estimate:
            An ndarray of PSD values to normalize.
        freqs:
            A 1-D array of frequency values at which PSD was estimated.
        start:
            The start frequency to begin measuring power at. If not in
            freqs, the closest value in freqs will be used. If None the
            first frequency in freqs is used.
        stop:
            The stop frequency to stop measuring power at. If not in
            freqs, the closest value in freqs will be used. If None the
            last frequency in freqs is used.
        axis: int
            The frequency axis of the psd ndarray.

    Returns:
        An array of normalized PSD values the same shape as estimate.

    Examples:
        >>> # Compute the PSD of the demo data
        >>> # import demo data and make a producer
        >>> from openseize.demos import paths
        >>> from openseize.file_io.edf import Reader
        >>> from openseize import producer
        >>> from openseize.spectra.estimators import psd
        >>> from openseize.spectra.metrics import power_norm
        >>> fp = paths.locate('recording_001.edf')
        >>> reader = Reader(fp)
        >>> pro = producer(reader, chunksize=10e4, axis=-1)
        >>> # Compute the PSD
        >>> n, freqs, estimate = psd(pro, fs=5000, axis=-1)
        >>> # compute the power in the 0-40Hz for each channel
        >>> result = power_norm(estimate, freqs, start=0, stop=40)
        >>> print(result.shape)
        (4, 5001)
    """

    norm = power(estimate, freqs, start, stop, axis)
    norm = np.expand_dims(norm, axis=axis)
    return estimate / norm


def confidence_interval(psd: npt.NDArray[np.float64],
                        n_estimates: int,
                        alpha: float = 0.05
) -> List[Tuple[float, float]]:
    """Returns the 1-alpha level confidence interval for each signal in
    a power spectrum estimate.

    Args:
        psd:
            An ndarray of signals to construct confidence intervals for.
        n_estimates:
            The number of segments used to construct the PSD estimate.
        alpha:
            A float in [0, 1) expressing the probability that a future
            interval will miss the True PSD. Default is 0.05 or a 95%
            CI.

    Returns:
        A list of lower and upper CI boundaries, one per signal in
         this estimators averaged PSD estimate.

    Interpretation:
        The confidence interval for the PS(D) defines an upper and lower
        bound in which we expect at 1-alpha % probability that future
        intervals will tend to contain the True PS(D). The narrower this
        interval, the more confident we can be that our estimate is "near"
        the True PS(D).

    References:

    - Shiavi, R. (2007). Introduction to Applied Statistical Signal
    Analysis : Guide to Biomedical and Electrical Engineering
    Applications. 3rd ed.
    """

    # degrees of freedom are the number of estimatives
    dof = n_estimates
    chi_bounds = chi2.ppf([alpha/2, 1-alpha/2], dof)

    # factor of 2 diff with Shiavi 7.48; 7.47 assumes complex signals
    lowers, uppers = [psd * dof / b for b in chi_bounds]

    return list(zip(lowers, uppers))
