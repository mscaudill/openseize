"""Tools for estimating the spectral content of ndarray data yielded by
a producer.

This module contains the following classes and functions:

    PSD:
        A function for estimating the power spectrum of a producer or
        ndarray using Welch's method.

    stft:
        A function for estimating the Short-time Fourier Transform of
        a producer or ndarray. 

    For Further implementation specific details please see
    openseize.numerical
"""

import numpy as np

from openseize.core import numerical as nm
from openseize.core.resources import is_assignable
from openseize import producer
from openseize.core.producer import Producer


def psd(data, fs, axis=-1, resolution=0.5, window='hann', overlap=0.5,
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
            The sampling rate of the data.
        axis: int
            The sample axis of the data. The estimate will be 
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


def stft(data, fs, axis, resolution=0.5, window='hann', overlap=0.5,
         detrend='constant', scaling='density', boundary=True, 
         padded=True, asarray=False):
    """A Short-Time Fourier Transform estimator.

    This estimator is useful for estimating changes in the frequency or 
    phase content of a non-stationary signal over time. 

    The STFT breaks a sequence of data into overlapping segments and
    computes a modified (windowed) Discrete Fourier Transform for each
    segment. This results in a sequence of complex values containing the
    frequency and phase content in each segment. These segments are then
    concatenated to create an estimate of the frequency content as
    a function of time X(frequncy, time). This is the STFT estimate.

    Args:
        data: An ndarray or producer of ndarrays
            A data producer whose stft is to be estimated.
        fs: int
            The sampling rate of the data.
        axis: int
            The sample axis of the data. The estimate will be 
            carried out along this axis.
        resolution: float
            The frequency resolution of the estimate in Hz for each segment.
            The resolution determines the number of DFT frequencies between
            [0, fs) to use to estimate the power spectra. The number of DFT
            points is fs // resolution. The default resolution is 0.5 Hz.
        window: str
            A string name for a scipy window to be applied to each data
            segment before computing the modified DFT of that segment. For
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
            'spectrum' the modulus of the estimate squared will have units
            V**2 for each segment and if density the modulus squared of the
            estimate for each segment will have units V**2/Hz. Default scaling
            is 'density'.
        boundary: bool
            A boolean indicating if data should be padded along axis at both
            ends with nfft//2 zeros. Here nfft is the number of dft points
            in a segment (fs/resolution). If True, the endpts of the signal 
            will recieve the same window weight as all other points allowing
            for accurate reconstruction of the signal from the stft estimate
            through inversion (istft not implemented). Default is True.
        padded: bool
             boolean indicating if the signal should be extended with zeros
             so that an integer number of windows covers the signal. This
             ensures that all of the signal is used in the estimate and can
             be recovered via inversion. In contrast PSD drops the last
             segment if less than nfft in size. Default is True.
        asarray: bool
            Boolean indicating if the estimator should attempt to return the
            result as an ndarry. If False the returned result will be
            a producer that produces segements of the STFT estimate. Default
            is False.
            
    Notes:
        Scipy allows for non-zero boundary padding. Since zero extension has
        the simple interpretation of frequency interpolation of the FFT,
        openseize only allows for zero extensions of the boundaries.

    Returns: 
        freqs: A 1-D array of frequencies at the requested resolution.
        time: A 1-D array of times for the segments used in the estimate.
        X: A producer of STFT segments. Stacking these estimates along last
           axis will create the STFT estimate with segment times along last 
           axis. The length of each produced array along axis will be 
           nfft - overlap * nfft where nfft = fs // resolution.

    References:
    (1) Oppenheim, Alan V., Ronald W. Schafer, John R. Buck “Discrete-Time 
        Signal Processing”, Prentice Hall, 1999.
    (2) B. Porat, "A Course In Digitial Signal Processing" Chapters 
        4 & 13. Wiley and Sons 1997.
    """

    pro = producer(data, chunksize=fs, axis=axis)
    
    # convert requested resolution to DFT pts
    nfft = int(fs / resolution)

    freqs, time, result = nm.stft(pro, fs, nfft, window, overlap, axis,
                                    detrend, scaling, boundary, padded)

    # attempt to return ndarray if requested
    if asarray:
        
        if is_assignable(result):
            result = np.stack([arr for arr in result], axis=-1)
    
    return freqs, time, result

