import abc
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import signaltools, firwin, freqz, convolve, kaiserord
from scipy.signal.windows import kaiser
from matplotlib.patches import Rectangle

from openseize.mixins import ViewInstance
from openseize.filtering.viewer import FilterViewer
from openseize.tools.numtools import _oa_array

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

    def apply(self, signal, axis, outtype, mode):
        """Apply this FIR to an ndarray of data.

        Args:
            arr (ndarray):      array with samples along last axis
                                or a generator!! with chunksize

            #FIXME
            phase_shift (bool): whether to compensate for the filter's
                                phase shift (Default is True)
        
        Returns: ndarry of filtered signal values
        """

        # possible inputs/outputs
        # 1. array
        # 2. a generator
        # Need to validate these type strings

        if isinstance(signal, np.ndarray):
            #we may have a memmap or in-memory array
            if outtype == 'array':
                #if using, use axis arg to set newaxis correctly
                axes = list(range(signal.ndim))
                axes.pop(axis)
                x = np.expand_dims(self.coeffs, axes)
                result = convolve(signal, x, mode=mode)
            if outtype == 'generator':
                #call oa algorithm
                result = _oa_array(signal, self.coeffs, axis=axis, mode=mode)
        else:
            result = self.overlap_add(signal)
        return result
            







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
    
    import time

    time_s = 10
    fs = 5000
    nsamples = int(time_s * fs)
    t = np.linspace(0, time_s, nsamples)

    # make a small 10 Hz riding on top of a larger 100 Hz sinusoidal signal
    x = 0.5 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t) + \
            0.25*np.random.random(nsamples)
    y = 2 * np.sin(2 * np.pi * 10 * t) + 0.5 *np.random.random(nsamples) 
    z = 2 * np.cos(2 * np.pi * 7.5 * t) + 0.5 *np.random.random(nsamples) 

    arr = np.stack((x,y, z, 1.5*x, y))
    #make ndarray
    arr2 = np.stack((arr, arr, arr))
        
    fir = Kaiser(50, width=20, fs=5000, btype='lowpass', 
                pass_ripple=.005, stop_db=40)
  
    
    t0 = time.perf_counter()
    a = fir.apply(arr, outtype='array', axis=-1, mode='full')
    print('filtered in {}s'.format(time.perf_counter() - t0))
    plt.ion()
    fig, axarr = plt.subplots(5,1)
    [axarr[idx].plot(row) for idx,row in enumerate(arr)]
    [axarr[idx].plot(row, color='r') for idx, row in enumerate(a)]
    plt.show()
   
    t0 = time.perf_counter()
    g = fir.apply(arr, outtype='generator', axis=-1, mode='full')
    b = np.concatenate([arr for arr in g], axis=1)
    print('filtered in {}s'.format(time.perf_counter() - t0))
    plt.ion()
    fig, axarr = plt.subplots(5,1)
    [axarr[idx].plot(row, color='g') for idx,row in enumerate(arr)]
    [axarr[idx].plot(row, color='tab:orange') for idx, row in enumerate(b)]
    plt.show()
    

    """
    a = fir.apply(arr2, outtype='array', axis=-1, mode='full')
    gen = fir.apply(arr2, outtype='generator', axis=-1, mode='full')
    b = np.concatenate([arr for arr in gen], axis=-1)
    """
    

    print(np.allclose(a,b))
