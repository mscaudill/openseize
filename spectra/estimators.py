"""Tools for estimating the spectral content of ndarray data yielded by
a producer.

This module contains the following classes and functions:

    PSD:
        A function for estimating the power spectrum of a producer using
        Welch's method.

    normalize:
        A function that normalizes an array of PSDs by the total power in
        a band (see spectra.metrics.power)

    For Further implementation specific details please see
    openseize.numerical.welch and scipy.signal.welch
"""

import numpy as np
import matplotlib.pyplot as plt


from openseize.core import numerical as nm
from openseize import producer
from openseize.core.producer import Producer
from openseize.spectra.plotting import banded
from openseize.spectra.metrics import power


def PSD(data, fs, axis=-1, resolution=0.5, window='hann', overlap=0.5,
        detrend='constant', scaling='density'):
    """A power spectrum (density) estimator using Welch's method.

    Welch's method divides data into overlapping segments and averages the
    modified periodograms for each segment. This estimator can process 
    ndarrays or producer of ndarrays allowing it to estimate the PSD for
    large data sets.

    Args:
        data: An ndarray or producer of ndarrays
            A data producer whose power spectral is to be estimated.
        fs: int
            The sampling rate of the produced data.
        axis: int
            The sample axis of the producer. The estimate will be 
            carried out along this axis.
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
        
    Returns: The integer number of windows averaged to compute the PSD 
             estimate; a 1D array of frequencies at which the PSD was 
             estimated and the average PSD estimate ndarray.

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

    pro = producer(data, chunksize=fs, axis=axis)
    
    # convert requested resolution to DFT pts
    nfft = int(fs / resolution)

    # build a producer of psd estimates, one per welch segment & store
    freqs, psd_pro = nm.welch(pro, fs, nfft, window, overlap, axis, detrend,
                              scaling)

    # compute the average PSD estimate
    result = 0
    for cnt, arr in enumerate(psd_pro, 1):
        result = result + 1 / cnt * (arr - result)
    
    return cnt, freqs, result


def normalize(estimate, freqs, start=None, stop=None, axis=-1):
    """Normalizes power spectral densities by the total power between 
    start and stop frequencies.

    Args:
        estimate: ndarray
            An ndarray of PSD values to normalize.
        freqs: 1-D array
            Array of frequency values at which PSD was estimated.
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
        An array of normalized PSD values.
    """
        
    norm = power(estimate, freqs, start, stop, axis)
    norm = np.expand_dims(norm, axis=axis)
    return estimate / norm



if __name__ == '__main__':

    from openseize.io import edf
    from openseize.demos import paths
    from openseize.spectra.metrics import confidence_interval
    import time


    def fetch_data(start, stop):
        """ """

        fp = paths.locate('recording_001.edf')
        with edf.Reader(fp) as reader:
            arr = reader.read(start, stop)

        return arr
    
    data = fetch_data(0, 200000)

    t0 = time.perf_counter()
    n, freqs, avg_psd = PSD(data, fs=5000, axis=-1) 
    norm_psd = normalize(avg_psd, freqs, start=0, stop=None)
    print('openseize estimated in {} s'.format(time.perf_counter() - t0))

    ci_s = confidence_interval(norm_psd, n, alpha=0.05)

    fig, axarr = plt.subplots(3, 1, sharex=True, figsize=(5,6))
    for idx, ax in enumerate(axarr):
        lower, upper = ci_s[idx]
        ax.plot(freqs, norm_psd[idx], color='green', label='Avg Norm PSD')
        ax = banded(freqs, upper, lower, ax, label='95% CI')
        ax.legend()

    plt.xlim([-1, 20])
    title = 'Data Duration = {} s'.format(data.shape[-1] / 5000)
    plt.suptitle(title)
    plt.show()
