"""A module for testing specialized FIR and IIR filters.

Typical ussage example:
    !pytest test_special.py::<TEST NAME>
"""

import pytest
import numpy as np
import scipy.signal as sps

from openseize.filtering.special import Hilbert

def sine(amplitude=1, freq=8, phase=0, duration=10, fs=250):
    """Returns a sine wave with specified amplitude, frequency, phase and
    duration.

    Args:
        amplitude:
            Amplitude of the sine wave. Default is 1.
        freq:
            The frequency of the sine wave in Hz. Default is 8 hz.
        phase:
            The phase of the sine wave in degrees. Default is 0.
        duration:
            The duration of the sine wave in seconds. Default is 10 seconds.
        fs:
            The sampling rate. Default is 250 Hz.

    Returns:
        A tuple of times and a 1-D array of sine wave amplitudes.
    """

    # convert phase from degs to radians
    phi = phase * np.pi / 180
    times = np.arange(0, fs *(duration + 1/fs)) / fs
    return times, amplitude * np.sin(2 * np.pi * freq * times + phi)

def test_hilbert():
    """Test if Openseize's iterative Hilbert is close to the Scipy's exact
    solution.
    """

    fs = 250
    chunksize=500
    times, signal = sine(duration=30, fs=fs)

    # compute openseize hilbert transform
    hilbert = Hilbert(width=4, fs=fs)
    imag_comp = hilbert(signal, chunksize=chunksize, axis=-1)

    # compute scipy analytic signal and get imag component
    analytic = sps.hilbert(signal)
    sps_comp = np.imag(analytic)

    # Edge effects impact openseize's hilbert so drop first and last chunks
    x = imag_comp[chunksize:-chunksize]
    y = sps_comp[chunksize:-chunksize]

    # compare percent diffs
    percent_diff = np.abs(x-y)/np.abs(y)

    # assert that max percent difference of imaginary magnitude < 3%
    assert(np.max(percent_diff) < .03)
