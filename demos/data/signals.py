import numpy as np

def sine(freq, fs, duration, phase):
    """Creates a pure tone sine wave 
    
    Args:
        freq: float
            The frequency of the sine wave in Hz.
        fs: int
            The sampling rate of the sine wave  in Hz.
        duration: 
            The length of the sine wave in seconds.
        phase:
            The phase of the sine wave in degrees.

    Returns: An array of length fs * duration of sine wave values.
    """

    t = np.linspace(0, duration, fs*duration)

