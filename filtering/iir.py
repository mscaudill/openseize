import numpy as np
import scipy.signal as sps

from openseize.filtering.bases import IIR

class Butter(IIR):
    """ """

    def __init__(self, fpass, fstop, fs, gpass=1.5, gstop=40, fmt='sos'):
        """ """

        super().__init__(fpass, fstop, gpass, gstop, fs, fmt)

    @property
    def order(self):
        """ """

        n, wn =  sps.buttord(self.fpass, self.fstop, self.gpass, self.gstop, 
                             fs=self.fs)
        print(n, wn)
        return n, wn

if __name__ == '__main__':

    time = 10
    fs = 5000
    nsamples = int(time * fs)
    t = np.linspace(0, time, nsamples)

    # make a small 10 Hz riding on top of a larger 100 Hz sinusoidal signal
    x = 0.5 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t) + \
            0.25*np.random.random(nsamples)
    y = 2 * np.sin(2 * np.pi * 22 * t) + 0.5 *np.random.random(nsamples) 

    arr = np.stack((x, y))

    butter = Butter(fpass=[400], fstop=[500], fs=fs) #lowpass
    #butter = Butter(fpass=[500], fstop=[400], fs=fs) #highpass
    #butter = Butter(fpass=[500, 700], fstop=[400, 800], fs=fs) #bandpass

