import abc
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import signaltools, firwin, freqz, convolve, kaiserord
from scipy.signal.windows import kaiser
from matplotlib.patches import Rectangle

from openseize.mixins import ViewInstance
from openseize.filtering.viewer import FilterViewer

class FIR(abc.ABC, ViewInstance, FilterViewer):
    """Abstract Finite Impulse Response Filter defining required and common
    methods used by all FIR subclasses.

    Attrs:
        fs (int):                       sampling frequency in Hz
        nyq (int):                      nyquist frequency
        cutoff (float or 1-D array):    freqs at which Gain of the filter
                                        drops to <= -6dB
        width (int):                    width of transition bewteen pass and
                                        stop bands
        btype (str):                    type of filter must be one of
                                        {lowpass, highpass, bandpass, 
                                        bandstop}. Default is lowpass
        pass_ripple (float):            the maximum deviation in the pass
                                        band (Default=0.005 => 0.5% ripple)
        stop_db (float):                minimum attenuation to achieve at
                                        end of transition width in dB 
                                        (Default=40 dB ~ 99% amplitude
                                        reduction)
        -- Computed --
        ntaps (int):                    number filter taps to achieve pass,
                                        stop and transition width criteria
        coeffs (arr):                   filter coeffs of the designed filter
                                        "h(n)"
        others:                         Each FIR may add additional computed
                                        attrs
    """

    def __init__(self, cutoff, width, fs, btype='lowpass', pass_ripple=0.005, 
                 stop_db=40):
        """Initialize this FIR filter."""

        self.fs = fs
        self.nyq = fs/2
        self.cutoff = np.atleast_1d(cutoff)
        self.norm_cutoff = self.cutoff / self.nyq
        self.width = width
        self.btype = btype
        self.pass_ripple = pass_ripple
        self.stop_db = stop_db

    @abc.abstractmethod
    def _build(self):
        """Returns this FIR filters coefficients in a 'coeffs' attr."""

    def _count_taps(self, phasetype=1):
        """Returns the Bellanger estimate of the number of taps.

        Scipy does not automatically provide the number of taps needed to
        build a filter that meets the width, pass_ripple and stop_db
        attenuation requirements for filters other than Kaiser. This method
        provides an estimate of the number of taps needed for a general FIR
        filter and is suited for use with optimal filter design methods such
        as Scipy's remez. Remez will be implemented at a future date.
    
        Args:
            width (float):              width of transition between pass and
                                        stop bands of the filter
            fs (int):                   sampling rate in Hz
            pass_ripple (float):        maximum deviation in the pass
                                        band (Default=0.005 => 0.5% ripple)
            stop_db (float):            minimum attenuation to achieve at
                                        end of transition width in dB 
                                        (Default=40 dB ~ 99% gain reduction)
            phasetype (int):            linear phase type must be one of 
                                        (1,2,3,4), (Default=1)

        Returns: integer number of types

        Ref: Ballenger (2000). Digital Processing of signals: Theory and 
             Practice 3rd ed. Wiley
        """

        stop_gain = 10 ** (-self.stop_db / 20)
        ntaps = -2/3 * np.log10(10 * self.pass_ripple * stop_gain) * (
                self.fs / self.width)
        if phasetype % 2 == 1:
            #odd phase type -> ensure number of taps is odd
            ntaps = ntaps + 1 if ntaps % 2 == 0 else ntaps
        else:
            #even phase type -> ensure number of taps is even
            ntaps = ntaps + 1 if ntaps % 2 == 1 else ntaps
        return ntaps

    def apply(self, arr, phase_shift=True):
        """Apply this FIR to an ndarray of data.

        Args:
            arr (ndarray):      array with samples along last axis
            phase_shift (bool): whether to compensate for the filter's
                                phase shift (Default is True)
        
        Returns: ndarry of filtered signal values
        """

        #FIXME USE ADD OVERLAP RETURNING GENERATOR FOR LARGE DATA
        mode = 'same' if phase_shift else 'full'
        return convolve(arr, self.coeffs[np.newaxis,:], mode=mode)


class Kaiser(FIR):
    """A Type I Finitie Impulse Response filter constructed using the Kaiser
    window method.

    Attrs:
        fs (int):                       sampling frequency in Hz
        nyq (int):                      nyquist frequency
        cutoff (float or 1-D array):    freqs at which Gain of the filter
                                        drops to <= -6dB
        width (int):                    width of transition bewteen pass and
                                        stop bands
        btype (str):                    type of filter must be one of
                                        {lowpass, highpass, bandpass, 
                                        bandstop}. Default is lowpass
        pass_ripple (float):            the maximum deviation in the pass
                                        band (Default=0.005 => 0.5% ripple)
        stop_db (float):                minimum attenuation to achieve at
                                        end of transition width in dB 
                                        (Default=40 dB ~ 99% amplitude
                                        reduction)
        -- Computed --
        ntaps (int):                    number filter taps to achieve pass,
                                        stop and transition width criteria
        beta (float):                   kaiser window shape parameter (see
                                        scipy kasier window)
        coeffs (arr):                   filter coeffs of the designed filter
                                        "h(n)"

    Scipy's firwin requires the number of taps to determine the transition 
    width between pass and stop bands. FIR_I uses the transition width and 
    the attenuation criteria to determine the number of taps automatically.

    References:
        1. Ifeachor E.C. and Jervis, B.W. (2002). Digital Signal Processing:
           A Practical Approach. Prentice Hall
        2. Oppenheim, Schafer, "Discrete-Time Signal Processing".
    """

    def __init__(self, cutoff, width, fs, btype='lowpass', 
                 pass_ripple=0.005, stop_db=40):
        """Initialize and build this FIR filter."""

        super().__init__(cutoff, width, fs, btype=btype, 
                         pass_ripple=pass_ripple, stop_db=stop_db)
        self.ntaps, self.beta = self._count_taps()
        self.coeffs = self._build()

    def _count_taps(self):
        """Returns the minimum number of taps needed for this FIR's
        attenuation and transition width criteria with a Kaiser window.

        Oppenheim, Schafer, "Discrete-Time Signal Processing", pp.475-476.
        """

        #find most restrictive dB criteria
        pass_db = -20 * np.log10(self.pass_ripple)
        design_param = max(pass_db, self.stop_db)
        #compute taps and shape parameter
        ntaps, beta = kaiserord(design_param, self.width/(self.nyq))
        #Symmetric FIR type I requires odd tap num
        ntaps = ntaps + 1 if ntaps % 2 == 0 else ntaps
        return ntaps, beta

    def _build(self):
        """Build & return the Kaiser windowed filter."""

        #call scipy firwin returning coeffs
        return firwin(self.ntaps, self.norm_cutoff, window=('kaiser', 
                       self.beta), pass_zero=self.btype, scale=False)


    
        
if __name__ == '__main__':

    time = 10
    fs = 5000
    nsamples = int(time * fs)
    t = np.linspace(0, time, nsamples)

    # make a small 10 Hz riding on top of a larger 100 Hz sinusoidal signal
    x = 0.5 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t) + \
            0.25*np.random.random(nsamples)
    y = 2 * np.sin(2 * np.pi * 10 * t) + 0.5 *np.random.random(nsamples) 

    arr = np.stack((x,y, 1.5*x, y))

        
    fir = Kaiser([100, 200], width=30, fs=1000, btype='bandpass', 
                pass_ripple=.005, stop_db=40)
    fir.view()

    """
    results = fir.apply(arr)

    fig, axarr = plt.subplots(4,1)
    [axarr[idx].plot(row) for idx,row in enumerate(arr)]
    [axarr[idx].plot(row, color='r') for idx, row in enumerate(results)]
    plt.show()
    """
