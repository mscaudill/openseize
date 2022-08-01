import numpy as np

from scipy.integrate import simpson
from scipy.stats import chi2

from openseize.core.arraytools import nearest1D, slice_along_axis


def power(psd, freqs, start=None, stop=None, axis=-1):
    """Returns the power in an array of power spectrum densities between
    start and stop frequencies using Simpson's rule.

    Args:
        psd: ndarray
            An array of power spectral densities.
        freqs: 1-D array
            An array of frequencies for each psd value along axis.
        start: float
            The start frequency to begin measuring power at. If not in
            freqs, the closest value in freqs will be used. If None the
            first frequency in freqs is used. Default is None.
        stop: float
            The stop frequency to stop measuring power at. If not in
            freqs, the closest value in freqs will be used. If None the
            last frequency in freqs is used. Default is None.
        axis: int
            The frequency axis of the psd ndarray.

    Returns:
        An array with one fewer dimension than PSD containing the powers
        measured along axis.

    Note: This metric should only be used on data that is density scaled.
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
    
    return result


def confidence_interval(psd, n, alpha=0.05):
        """Returns the 1-alpha level confidence interval for each signal in 
        a power spectrum estimate.

        Args:
            psd: ndarray
                An ndarray of signals to construct confidence intervals for.
            n: int
                The number of segments used to construct the PSD estimate.
            alpha: float in [0, 1)
                Alpha expresses the probability that a future interval will
                miss the True PSD. Default is 0.05 or a 95% CI.

        Returns: list of lower and upper CI boundaries, one per signal in
        this estimators averaged PSD estimate.
        
        Interpretation:
        The confidence interval for the PS(D) defines an upper and lower
        bound in which we expect at 1-alpha % probability that future
        intervals will tend to contain the True PS(D). The narrower this 
        interval, the more confident we can be that our estimate is "near" 
        the True PS(D).

        References:
        Shiavi, R. (2007). Introduction to Applied Statistical Signal 
        Analysis : Guide to Biomedical and Electrical Engineering 
        Applications. 3rd ed.
        """
        
        # degrees of freedom are the number of estimatives
        dof = n
        chi_bounds = chi2.ppf([alpha/2, 1-alpha/2], dof)

        # factor of 2 diff with Shiavi 7.48; 7.47 assumes complex signals
        lowers, uppers = [psd * dof / b for b in chi_bounds]

        return list(zip(lowers, uppers))

