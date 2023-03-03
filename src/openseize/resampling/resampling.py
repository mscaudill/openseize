"""Tools for downsampling, upsampling & resampling data using polyphase
decompositions of the FIR antialiasing & interpolation filter coeffecients.

This module contains the following functions:

**downsample:**
A function that downsamples an array or producer of arrays by an
integer decimation factor in conjunction with an antialiasing filter.

Examples:
    >>> # Get demo data and build a reader then a producer
    >>> from openseize.demos import paths
    >>> filepath = paths.locate('recording_001.edf')
    >>> from openseize.io.edf import Reader
    >>> reader = Reader(filepath)
    >>> pro = producer(reader, chunksize=100e3, axis=-1)
    >>> # downsample the data from 5 kHz to 500 Hz
    >>> downpro = downsample(pro, M=10, fs=5000, chunksize=100e3)
    >>> print(pro.shape)
    (4, 18875000)
    >>> x = downpro.to_array() # convert downsampled to an array
    >>> print(x.shape)
    (4, 1887500)

___
**upsample:**
A function that upsamples an array or producer of arrays by an
integer expansion factor in conjunction with an interpolation filter.

Examples:
    >>> # use previous downsampled array x and upsample by 3
    >>> # upsample x from 500 Hz to 1500 Hz
    >>> up_data = downsample(x, L=3, fs=500, chunksize=100e3)
    >>> print(up_data.shape)
    (4, 5662500)

___
**resample:**
A function that resamples an array or producer of arrays by
a rational number L/M where L is the expansion factor and M is the
decimation factor. This allows resampling of digital signals at
non-integer frequency ratios.
    
Examples:
    >>> # Get demo data and build a reader then a producer
    >>> from openseize.demos import paths
    >>> filepath = paths.locate('recording_001.edf')
    >>> from openseize.io.edf import Reader
    >>> reader = Reader(filepath)
    >>> pro = producer(reader, chunksize=100e3, axis=-1)
    >>> #resample data from 5000 Hz to 1500 Hz
    >>> resample_pro = resample(pro, L=3, M=10, fs=5000,
    >>>                         chunksize=100e6)
    >>> y = resample_pro.to_array()
    >>> print(y.shape)
    (4, 5662500)

___
"""

from functools import partial
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from openseize.core.numerical import polyphase_resample
from openseize.core.producer import producer
from openseize.core.producer import Producer
from openseize.filtering.fir import Kaiser


def resampled_shape(pro: Producer,
                    L: int,
                    M: int,
                    axis: int) -> Tuple[int,...]:
    """Returns the resampled shape of a producer along axis.

    Args:
        pro:
            The data to be downsampled.
        L:
            The expansion factor. L-1 interpolated values will be inserted
            between consecutive samples in data along axis.
        M:
            The decimation factor describing which Mth samples of data
            survive decimation. (E.g. M=10 -> every 10th sample survives)
        axis:
            The axis of pro along which resampling will occur.

    Returns: shape tuple
    """

    shape = list(pro.shape)
    shape[axis] = int(np.ceil(pro.shape[axis] * L / M))
    return tuple(shape)


def downsample(data: Union[Producer, npt.NDArray[np.float64]],
               M: int,
               fs: float,
               chunksize: int,
               axis: int = -1,
               **kwargs,
) -> Union[Producer, npt.NDArray[np.float64]]:
    """Downsamples an array or producer of arrays using polyphase
    decomposition by an integer decimation factor.

    Args:
        data:
            The producer or ndarray data to be downsampled.
        M:
            The decimation factor describing which Mth samples of data
            survive decimation. (E.g. M=10 -> every 10th sample survives)
        fs:
            The sampling rate of data in Hz.
        chunksize:
            The number of samples to hold in memory during downsampling.
            This method will require ~ 3 times chunksize in memory.
        axis:
            The axis of data along which downsampling will occur.
        kwargs:
            Any valid keyword for a Kaiser lowpass filter. The default
            values for this antialiasing filter are:

            - fstop: int
                The stop band edge frequency.
                Defaults to cutoff + cutoff / 10 where cutoff =
                fs // (2 * M).
            - fpass: int
                The pass band edge frequency. Must be less than fstop.
                Defaults to cutoff - cutoff / 10 where cutoff = fs //
                (2 * M).
            - gpass: int
                The pass band attenuation in dB. Defaults to a max loss
                in the passband of 0.1 dB = ~1.1% amplitude loss.
            - gstop: int
                The max attenuation in the stop band in dB. Defaults to
                40 dB or 99%  amplitude attenuation.

    Returns:
        An array or producer of arrays of downsampled data depending on 
          input data parameters datatype.

    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 12 "Multirate Signal Processing"
        2. Polyphase implementation: scipy.signal.resample_poly
    """

   # no downsample requested
    if M == 1:
        return data

    pro = producer(data, chunksize, axis)

    # construct polyphase-resampling generating func & get resultant shape
    genfunc = partial(polyphase_resample, pro, 1, M, fs, Kaiser, axis,
                      **kwargs)
    shape = resampled_shape(pro, L=1, M=M, axis=axis)

    #build producer from generating function
    result = producer(genfunc, chunksize, axis, shape=shape)

    return result.to_array() if isinstance(data, np.ndarray) else result


def upsample(data: Union[Producer, npt.NDArray[np.float64]],
             L: int,
             fs: float,
             chunksize: int,
             axis: int = -1,
             **kwargs,
) -> Union[Producer, npt.NDArray[np.float64]]:
    """Upsamples an array or producer of arrays using polyphase
    decomposition by an integer expansion factor.

    Args:
        data:
            The producer or ndarray data to be downsampled.
        L:
            The expansion factor. L-1 interpolated values will be inserted
            between consecutive samples in data along axis.
        fs:
            The sampling rate of data in Hz.
        chunksize:
            The number of samples to hold in memory during upsampling.
            This method will require ~ 3 times chunksize in memory.
        axis:
            The axis of data along which upsampling will occur.
        kwargs:
            Any valid keyword for a Kaiser lowpass filter. The default
            values for this interpolation filter are:

            - fstop: int
                The stop band edge frequency.
                Defaults to cutoff + cutoff / 10 where cutoff =
                fs // (2 * L).
            - fpass: int
                The pass band edge frequency. Must be less than fstop.
                Defaults to cutoff - cutoff / 10 where cutoff = fs //
                (2 * L).
            - gpass: int
                The pass band attenuation in dB. Defaults to a max loss
                in the passband of 0.1 dB = ~1.1% amplitude loss.
            - gstop: int
                The max attenuation in the stop band in dB. Defaults to
                40 dB or 99%  amplitude attenuation.

    Returns:
        An array or producer of arrays of upsampled data depending on the 
          input data parameters datatype.

    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 12 "Multirate Signal Processing"
        2. Polyphase implementation: scipy.signal.resample_poly
    """

    # no upsample requested
    if L == 1:
        return data

    pro = producer(data, chunksize, axis)

    # construct polyphase-resampling generating func & get resultant shape
    genfunc = partial(polyphase_resample, pro, L, 1, fs, Kaiser, axis,
                      **kwargs)
    shape = resampled_shape(pro, L=L, M=1, axis=axis)

    #build producer from generating function
    result = producer(genfunc, chunksize, axis, shape=shape)

    return result.to_array() if isinstance(data, np.ndarray) else result


def resample(data: Union[Producer, npt.NDArray[np.float64]],
             L: int,
             M: int,
             fs: float,
             chunksize: int,
             axis: int = -1,
             **kwargs,
) -> Union[Producer, npt.NDArray[np.float64]]:
    """Resamples an array or producer of arrays using polyphase
    decomposition by a rational factor L / M.

    Args:
        data:
            The producer or ndarray of data to be resampled.
        L:
            The expansion factor. L-1 interpolated values will be inserted
            between consecutive samples in data along axis.
        M:
            The decimation factor describing which Mth samples of data
            survive decimation. (E.g. M=10 -> every 10th sample survives)
        fs:
            The sampling rate of data in Hz.
        chunksize:
            The number of samples to hold in memory during upsampling.
            This method will require ~ 3 times chunksize in memory.
        axis:
            The axis of data along which resampling will occur.
        kwargs:
            Any valid keyword for a Kaiser lowpass filter. The default
            values for combined antialiasing & interpolation filter are:

            - fstop: int
                The stop band edge frequency.
                Defaults to cutoff + cutoff / 10 where cutoff =
                fs / (2 * max(L,M)).
            - fpass: int
                The pass band edge frequency. Must be less than fstop.
                Defaults to cutoff - cutoff / 10 where cutoff = fs /
                (2 * max(L,M)).
            - gpass: int
                The pass band attenuation in dB. Defaults to a max loss
                in the passband of 0.1 dB = ~1.1% amplitude loss.
            - gstop: int
                The max attenuation in the stop band in dB. Defaults to
                40 dB or 99%  amplitude attenuation.

    Note:
        The resampling factor L/M may be reducible. If so, L and M are
        reduced prior to resampling.

    Returns:
        An array or producer of arrays of resampled data depending on input
          data parameters datatype.

    References:
        1. Porat, B. (1997). A Course In Digital Signal Processing. John
           Wiley & Sons. Chapter 12 "Multirate Signal Processing"
        2. Polyphase implementation: scipy.signal.resample_poly
    """

    # reduce the rational L/M
    g = np.gcd(L, M)
    l = L // g
    m = M // g

    # no resample requested
    if l == m == 1:
        return data

    pro = producer(data, chunksize, axis)
    # construct polyphase-resampling generating func & get resultant shape
    genfunc = partial(polyphase_resample, pro, l, m, fs, Kaiser, axis,
                      **kwargs)
    shape = resampled_shape(pro, L=L, M=M, axis=axis)

    #build producer from generating function
    result = producer(genfunc, chunksize, axis, shape=shape)

    return result.to_array() if isinstance(data, np.ndarray) else result
