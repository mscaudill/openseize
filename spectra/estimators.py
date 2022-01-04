import numpy as np
import scipy.signal as sps

from openseize.types.producer import producer

class Welch:
    """

    """

    def __init__(self, fs, nperseg, window='hann'):
        """Initialize this Welch."""
    
       # Method to view the Window and the FFT of the window with ENBW 
        

    def enbw(self):
        """ """
       
        return np.sum(self.window**2) / np.sum(self.window)**2 * self.fs
        

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftshift    

    fs = 5000
    window = sps.windows.hann(16384*2)
    #window = np.ones(10000)
    A = fft(window, 10000) / (len(window) / 2)
    freq = np.linspace(-fs/2, fs/2, len(A))
    resp = 20 * np.log10(np.abs(fftshift(A/ abs(A).max())))

    resolution =  np.sum(window**2) / np.sum(window)**2 * len(window)
    print(resolution)

    plt.plot(freq, resp)
    plt.show()
    plt.ion()
