"""Tools for estimating the spectral content of ndarray data yielded by
a producer.

This module contains the following classes and functions:

    PSD:
        A class for estimating the power spectrum of a producer using
        Welch's method.

        Typical usage example:
        
        # create an estimator of the power spectrum
        estimator = PSD(producer, fs=sampling_rate)
        
        # estimate method the average PS(D) estimate across all overlapping
        # segements of producer's data.
        freqs, psd = estimator.estimate()
        
        # compute the confidence iterval of the 0th signal in the estimate
        CI = estimator.confidence_interval(psd[0])

    For Further implementation specific details please see
    openseize.numerical.welch and scipy.signal.welch
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2

from openseize.core import numerical as nm
from openseize import producer
from openseize.core.producer import Producer
from openseize.spectra.plotting import banded


class PSD:
    """A power spectrum (density) estimator using Welch's method.

    Welch's method divides data into overlapping segments and averages the
    modified periodograms for each segment. This estimator 


    References:
        (1) P. Welch, "The use of the fast Fourier transform for the 
        estimation of power spectra: A method based on time averaging
        over short, modified periodograms", IEEE Trans. Audio 
        Electroacoust. vol.15, pp. 70-73, 1967

        (2) M.S. Bartlett, "Periodogram Analysis and Continuous 
        Spectra", Biometrika, vol. 37, pp. 1-16, 1950.

        (3) B. Porat, "A Course In Digitial Signal Processing" Chapters 
        4 & 13. Wiley and Sons 1997.

    """

    def __init__(self, data, fs, axis=-1):
        """Initialize this Estimator.

        Args:
            data: An ndarray or producer of ndarrays
                A data producer whose power spectral is to be estimated.
            fs: int
                The sampling rate of the produced data.
            axis: int
                The sample axis of the producer. The estimate will be 
                carried out along this axis.
        """

        self.pro = data
        if isinstance(data, np.ndarray):
            # chunksize is arbitrary since welch will set it
            self.pro = producer(data, chunksize=fs, axis=axis)

        self.fs = fs
        self.axis = axis
    
    def estimate(self, resolution=0.5, window='hann', overlap=0.5,
                 detrend='constant', scaling='density'):
        """Estimates the power spectrum (density) along axis of this
        Estimator's producer. 

        resolution: float
            The frequency resolution of the estimate in Hz. The resolution
            determines the number of DFT frequencies between [0, fs) to use
            to estimate the power spectra. The number of DFT points is
            fs // resolution. The default resolution is 0.5 Hz.
        window: str
            A string name for a scipy window to be applied to each data
            segment before computing the periodogram of that segment. For
            a full list of windows see scipy.signal.windows. The default
            window is the Hann window.
        overlap: float
            A percentage in [0, 1) of the data segement that should overlap
            with the next data segment. If 0 this estimate is equivalent to
            Bartletts method (2). The default overlap is 0.5.
        detrend: str either 'constant' or 'linear'
            A string indicating whether to subtract the mean ('constant') or
            subtract a linear fit ('linear') from each segment prior to
            computing the estimate for a segment. Default is 'constant'.
        scaling: str either 'spectrum' or 'density'
            Determines the normalization to apply to the estimate. If
            'spectrum' the estimate will have units V**2 and if 'density'
            V**2 / Hz. Default scaling is 'density'.
        
        Returns: A 1D array of frequencies at which the PSD was estimated
        and the average PSD estimate across all overlapping windows.

        Stores: The 1D array of frequencies and a producer that yields the
        PS(D) estimate for each overlapping segment.
        """

        # convert requested resolution to DFT pts
        nfft = int(self.fs / resolution)

        # build a producer of psd estimates, one per welch segment & store
        freqs, psd_pro = nm.welch(self.pro, self.fs, nfft, window, overlap,
                                  self.axis, detrend, scaling)
        self.freqs, self.psd_pro = freqs, psd_pro

        # compute the average PSD estimate
        result = 0
        for cnt, arr in enumerate(self.psd_pro, 1):
            result = result + 1 / cnt * (arr - result)
        self.avg_psd = result
        
        return freqs, result

    def confidence_interval(self, alpha=0.05):
        """Returns the 1-alpha level confidence interval for each signal in 
        this estimate.

        Args:
            alpha: float in [0, 1)
                Alpha expresses the probability that a future interval will
                miss the True PSD.

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

        # degrees of freedom is avg segments
        dof = self.psd_pro.shape[self.axis]
        
        chi_bounds = chi2.ppf([alpha/2, 1-alpha/2], dof)

        # factor of 2 diff with Shiavi 7.48; 7.47 assumes complex signals
        lowers, uppers = [self.avg_psd * dof / b for b in chi_bounds]

        return list(zip(lowers, uppers))



if __name__ == '__main__':

    from openseize.io import edf
    from openseize.demos import paths


    def fetch_data(start, stop):
        """ """

        fp = paths.locate('recording_001.edf')
        with edf.Reader(fp) as reader:
            arr = reader.read(start, stop)

        return arr
    
    data = fetch_data(0, 500000)
    pro = producer(data, chunksize=20000, axis=-1)

    estimator = PSD(pro, fs=5000)
    freqs, avg_psd = estimator.estimate()

    ci_s = estimator.confidence_interval(0.05)

    fig, axarr = plt.subplots(3, 1, sharex=True)
    for idx, ax in enumerate(axarr):
        lower, upper = ci_s[idx]
        ax.plot(freqs, avg_psd[idx], color='green', label='Avg PSD')
        ax = banded(freqs, upper, lower, ax, label='95% CI')
        ax.legend()

    plt.show()
