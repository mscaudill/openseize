"""A collection of callable Finite Impulse Response filters.

This collection includes the Kaiser parametric filter and non-parametric 
general cosine window filters including. Each filter inherits the ability 
to plot itself and can be called on an ndarray or a producer of ndarrays.
Please see the FIR base class in openseize.filtering.bases for inherited
properties and methods.

Typical usage example:
    
    # build a lowpass Kaiser FIR with and edge frequencies of 500 and 600 Hz
    # that has a maximum passband loss of 1 dB ~ 11 % & a minimum stop band
    # attenuation of 40 dB ~ 99% amplitude attenuation.
    kfir = Kaiser(fpass=500, fstop=600, fs=5000, gpass=1.0, gstop=40)

    # plot this filter's impulse and frequency response
    kfir.plot()

    # finally apply the filter to a producer of ndarrays called 'data'.
    # and account for the group delay of the filter with mode='same'
    filtered = kfir(data, chunksize=10e6, axis=-1, mode='same')

    # process each filtered chunk
    for arr in filtered:
        # do something with filtered chunk 
"""

import numpy as np
import scipy.signal as sps

from openseize.filtering.bases import FIR

class Kaiser(FIR):
    """A callable Type I FIR Filter using a Kaiser window. 

    Designs a Kaiser windowed filter that meets the stricter of the pass and
    stop band attenuation criteria.

    Attrs:
        fpass, fstop: float or 1-D array
            Pass and stop band edge frequencies (Hz).
            For example:
                - Lowpass: fpass = 1000, fstop = 1100
                - Highpass: fpass = 2000, fstop = 1800 
                - Bandpass: fpass = [400, 800], fstop = [300, 900]
                - Bandstop: fpass = [100, 200], fstop = [120, 180]
        fs: int
            The sampling rate of the data to be filtered.
        gpass: float
            The maximum loss in the pass band (dB). Default is 1.0 dB which
            is an amplitude loss of ~ 11%.
        gstop: float
            The minimum attenuation in the stop band (dB). The default is 40
            dB which is an amplitude loss of 99%

    """

    def __init__(self, fpass, fstop, fs, gpass=1.0, gstop=40, **kwargs):
        """Initialize this Kaiser windowed FIR filter."""

        super().__init__(fpass, fstop, gpass, gstop, fs, **kwargs)

    @property
    def numtaps(self):
        """Returns the number of taps needed to meet the stricter of the
        pass and stop band criteria and the kaiser window parameter beta.

        The gpass attr of this FIR is the max loss in the passband. This is
        not the same as the attenuation in the pass band so we first compute
        the attenuation in the passband and compare against the attenuation
        in the stop band. The largest attenuation is then used to compute
        the minimum number of taps.
        """
        
        ripple = max(self.pass_attenuation, self.gstop)
        ntaps, _ = sps.kaiserord(ripple, self.width/self.nyq)
        return ntaps

    @property
    def window_params(self):
        """Returns the additional beta paramater required to create this 
        FIRs Kaiser window."""
        
        ripple = max(self.pass_attenuation, self.gstop)
        return [sps.kaiser_beta(ripple)]


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    fpass = [400, 800]
    fstop = [300, 900]

    kfir = Kaiser(fpass=fpass, fstop=fstop, gpass=1, gstop=40, fs=5000)

    w, h = sps.freqz(kfir.coeffs, fs=5000, worN=2000)
    fig, axarr = plt.subplots(2,1)
    axarr[0].plot(w, 20*np.log10(np.abs(h)))
    axarr[1].plot(w, np.abs(h))
    [ax.grid(alpha=0.5) for ax in axarr]
    [ax.axvline(fp, color='red') for fp in fpass for ax in axarr]
    [ax.axvline(fs, color='red') for fs in fstop for ax in axarr]
    [axarr[0].axvline([c], color='pink') for c in kfir.cutoff]
    [axarr[1].axvline([c], color='pink') for c in kfir.cutoff]
    plt.ion()
    plt.show()




       

