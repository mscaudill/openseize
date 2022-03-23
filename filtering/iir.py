import numpy as np
import scipy.signal as sps

from openseize.filtering.bases import IIR

class Butter(IIR):
    """ """

    def __init__(self, cutoff, width, fs, btype='lowpass', pass_db=1.5,
                 stop_db=40, fmt='sos'):
        """ """

        super().__init__(cutoff, width, fs, btype, pass_db, stop_db, fmt=fmt)

    @property
    def order(self):
        """ """

        wp, ws = self.edges()
        n, wn =  sps.buttord(wp, ws, self.pass_db, self.stop_db, fs=self.fs)
        print(wp, ws)
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

    butter = Butter(cutoff=500, width=20, fs=fs)

