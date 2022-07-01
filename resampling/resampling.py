"""Tools for downsampling, upsampling & resampling data using polyphase
decompositions of the FIR antialiasing & interpolation filter coeffecients.

This module contains the following functions:

    downsample:
        A function that downsamples an array or producer of arrays by an
        integer decimation factor in conjuction with an antialiasing filter.
        
        Typical usage example:
        pro = downsample(data, M=10, fs=5000, chunksize=10000, axis=-1)
        # returns a producer or ndarray of M-downsampled data where the
        downsampling axis is the last axis.

    upsample:
        A function that upsamples an array or producer of arrays by an
        integer expansion factor in conjuction with an interpolation filter.

        Typical usage example:
        pro = downsample(data, L=7, fs=5000, chunksize=10000, axis=-1)
        # returns a producer or ndarray of L-expanded data where the
        # expansion axis is the last axis. 

    resample:
        A function that resamples an array or producer of arrays by
        a rational number L/M where L is the expansion factor and M is the
        decimation factor. This allows resampling of digital signals at
        non-integer frequency ratios.
        Typical usage:
        # seek to downsample from frequency of 1000 Hz to 800 Hz
        pro = downsample(data, L=4, M=5, fs=1000, axis=-1)
        # returns an array or producer of arrays at sampling rate of 800 Hz.

For further details on implementation please see:
    openseize.numerical.polyphase_resample and scipy.signal.resample_poly
"""

import numpy as np

from openseize import producer
from openseize.core.numerical import polyphase_resample


def downsample(data, M, fs, chunksize, axis=-1, **kwargs):
    """Downsamples an array or producer of arrays using polyphase
    decomposition by an integer decimation factor.

    Args:
        data: ndarray or producer of ndarrays
            The data to be downsampled.
        M: int
            The decimation factor describing which Mth samples of data
            survive decimation. (E.g. M=10 -> every 10th sample survives)
        fs: int
            The sampling rate of data in Hz.
        chunksize: int
            The number of samples to hold in memory during downsampling.
            This method will require ~ 3 times chunksize in memory.
        axis: int
            The axis of data along which downsampling will occur.
        kwargs:
            Any valid keyword for a Kaiser lowpass filter. The default 
            values for this antialiasing filter are:

                fstop: int
                    The stop band edge frequency. Defaults to fs // M.
                fpass: int
                    The pass band edge frequency. Must be less than fstop.
                    Defaults to fstop - fstop // 10.
                gpass: int
                    The pass band attenuation in dB. Defaults to a max loss
                    in the passband of 1 dB ~ 11% amplitude loss.
                gstop: int
                    The max attenuation in the stop band in dB. Defaults to
                    40 dB or 99%  amplitude attenuation.
    
    Returns: An array or producer of arrays depending on data's datatype.
    
    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 12 "Multirate Signal Processing"
        2. Polyphase implementation: scipy.signal.resample_poly
    """
   
    pro = producer(data, chunksize, axis)
    result = polyphase_resample(pro, 1, M, fs, chunksize, axis, **kwargs)
   
    return result.to_array() if isinstance(data, np.ndarray) else result


def upsample(data, L, fs, chunksize, axis=-1, **kwargs):
    """Upsamples an array or producer of arrays using polyphase
    decomposition by an integer expansion factor.

    Args:
        data: ndarray or producer of ndarrays
            The data to be downsampled.
        L: int
            The expansion factor. L-1 interpolated values will be inserted
            between consecutive samples in data along axis.
        fs: int
            The sampling rate of data in Hz.
        chunksize: int
            The number of samples to hold in memory during upsampling.
            This method will require ~ 3 times chunksize in memory.
        axis: int
            The axis of data along which upsampling will occur.
        kwargs:
            Any valid keyword for a Kaiser lowpass filter. The default 
            values for this interpolation filter are:

                fstop: int
                    The stop band edge frequency. Defaults to fs // L.
                fpass: int
                    The pass band edge frequency. Must be less than fstop.
                    Defaults to fstop - fstop // 10.
                gpass: int
                    The pass band attenuation in dB. Defaults to a max loss
                    in the passband of 1 dB ~ 11% amplitude loss.
                gstop: int
                    The max attenuation in the stop band in dB. Defaults to
                    40 dB or 99%  amplitude attenuation.

    Returns: An array or producer of arrays depending on data's datatype.
    
    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 12 "Multirate Signal Processing"
        2. Polyphase implementation: scipy.signal.resample_poly
    """
    
    pro = producer(data, chunksize, axis)
    result = polyphase_resample(pro, L, 1, fs, chunksize, axis, **kwargs)
   
    return result.to_array() if isinstance(data, np.ndarray) else result


def resample(data, L, M, fs, chunksize, axis=-1, **kwargs):
    """Resamples an array or producer of arrays using polyphase 
    decomposition by a rational factor L / M.

    Args:
        data: ndarray or producer of ndarrays
            The data to be resampled.
        L: int
            The expansion factor. L-1 interpolated values will be inserted
            between consecutive samples in data along axis.
        M: int
            The decimation factor describing which Mth samples of data
            survive decimation. (E.g. M=10 -> every 10th sample survives)
        fs: int
            The sampling rate of data in Hz.
        chunksize: int
            The number of samples to hold in memory during upsampling.
            This method will require ~ 3 times chunksize in memory.
        axis: int
            The axis of data along which downsampling will occur.
        kwargs:
            Any valid keyword for a Kaiser lowpass filter. The default 
            values for combined antialiasing & interpolation filter are:

                fstop: int
                    The stop band edge frequency. 
                    Defaults to fs // max(L,M).
                fpass: int
                    The pass band edge frequency. Must be less than fstop.
                    Defaults to fstop - fstop // 10.
                gpass: int
                    The pass band attenuation in dB. Defaults to a max loss
                    in the passband of 1 dB ~ 11% amplitude loss.
                gstop: int
                    The max attenuation in the stop band in dB. Defaults to
                    40 dB or 99%  amplitude attenuation.

    Returns: An array or producer of arrays depending on data's datatype.

    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 12 "Multirate Signal Processing"
        2. Polyphase implementation: scipy.signal.resample_poly
    """

    pro = producer(data, chunksize, axis)
    result = polyphase_resample(pro, L, M, fs, chunksize, axis, **kwargs)
   
    return result.to_array() if isinstance(data, np.ndarray) else result
