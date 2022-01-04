import abc
import numpy as np
import scipy.signal as sps
from functools import partial

from openseize.core import mixins
from openseize.core.producer import Producer, producer
from openseize.core import numerical as onum
from openseize.filtering.viewer import FilterViewer

class FIR(abc.ABC, mixins.ViewInstance, FilterViewer):
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

    def _taps(self, phasetype=1):
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

    def apply(self, x, chunksize, axis, mode='same', **kwargs):
        """Apply this FIR to an ndarray or producer of ndarrays of data.

        Args:
            x  (producer or array-like):       an ndarray or producer of
                                               ndarrays of data to filter
            chunksize (int):                   sets the size of x to hold in
                                               memory per iteration
            axis (int):                        axis of x along which to
                                               apply the filter
            mode (str):                        boundary handling; one of
                                               {'full', 'same', 'valid'}
                                               These modes are identical to 
                                               numpy convolve. 
                                               'full': includes all points 
                                               of the convolution of the 
                                               filter with the data 
                                               including non-overlapping
                                               (zero-padded) endpts.
                                               'same': returns an array of
                                               of the same size as x
                                               'valid': returns an array
                                               where the filter and
                                               x completely overlap.
                    
        Returns: an ndarray or producer of ndarrays of filtered values
        """
     
        pro = producer(x, chunksize, axis, **kwargs)
        #build a producer of filtered values
        result = onum.oaconvolve(pro, self.coeffs, axis, mode)
        #if input is array -> return an array
        if isinstance(x, np.ndarray):
            result = np.concantenate([arr for arr in result], axis=axis)
        return result


class FIR_Kaiser(FIR):
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
        self.ntaps, self.beta = self._taps()
        self.coeffs = self._build()

    def _taps(self):
        """Returns the minimum number of taps needed for this FIR's
        attenuation and transition width criteria with a Kaiser window.

        Oppenheim, Schafer, "Discrete-Time Signal Processing", pp.475-476.
        """

        #find most restrictive dB criteria
        pass_db = -20 * np.log10(self.pass_ripple)
        design_param = max(pass_db, self.stop_db)
        #compute taps and shape parameter
        ntaps, beta = sps.kaiserord(design_param, self.width/(self.nyq))
        #Symmetric FIR type I requires odd tap num
        ntaps = ntaps + 1 if ntaps % 2 == 0 else ntaps
        return ntaps, beta

    def _build(self):
        """Build & return the Kaiser windowed filter."""

        #call scipy firwin returning coeffs
        return sps.firwin(self.ntaps, self.norm_cutoff, window=('kaiser', 
                       self.beta), pass_zero=self.btype, scale=False)


    
        
if __name__ == '__main__':
    
    import time
    import matplotlib.pyplot as plt
    from openseize.io.readers import EDF

    """
    time_s = 100
    fs = 5000
    nsamples = int(time_s * fs)
    t = np.linspace(0, time_s, nsamples)

    # make a small 10 Hz riding on top of a larger 100 Hz sinusoidal signal
    x = 0.5 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t) + \
            0.25*np.random.random(nsamples)
    y = 2 * np.sin(2 * np.pi * 10 * t) + 0.5 *np.random.random(nsamples) 
    z = 2 * np.cos(2 * np.pi * 7.5 * t) + 0.5 *np.random.random(nsamples) 

    arr = np.stack((x,y, z, 1.5*x, y))
        
    fir = Kaiser(50, width=100, fs=5000, btype='lowpass', 
                pass_ripple=.005, stop_db=40)

    t0 = time.perf_counter()
    pro = fir.apply(arr, chunksize=1000, axis=-1, mode='same')
    result = np.concatenate([arr for arr in pro], axis=-1)
    print('Openseize Filtering completed in {} s'.format(
          time.perf_counter() - t0))

    plt.ion()
    fig, axarr = plt.subplots(5,1)
    [axarr[idx].plot(row) for idx, row in enumerate(arr)]
    [axarr[idx].plot(row, color='r') for idx, row in enumerate(result)]

    #get result using scipy
    t0 = time.perf_counter()
    spresult = sps.convolve(arr, fir.coeffs[np.newaxis, :], mode='same')
    print('Scipy Filtering completed in {} s'.format(
          time.perf_counter() - t0))

    print('Filtered Equality? ', np.allclose(spresult, result))

    """

    PATH = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    edf = EDF(PATH)
    pro = producer(edf, chunksize=30e6, axis=-1)
    fir = Kaiser(500, width=100, fs=5000, btype='lowpass', 
                pass_ripple=.005, stop_db=40)

    result = fir.apply(pro, axis=-1, mode='same', chunksize=30e6)

    t0 = time.perf_counter()
    for idx, filt in enumerate(result):
        print(idx)
    print('FIR completed in {} s'.format(time.perf_counter() - t0))

    for idx, filt in enumerate(result):
        if idx == 0:
            fig, axarr = plt.subplots(4,1)
            [axarr[idx].plot(row[0:50000], color='r') for idx, row in enumerate(filt)]
        else:
            break
    #since loop above is terminated early manually reset chunksize 
    pro.chunksize = 30e6
    for idx, arr in enumerate(pro):
        if idx == 0:
            [axarr[idx].plot(row[0:50000], color='b', alpha=0.5) for idx, row in 
                    enumerate(arr)]
        else:
            break
    plt.show()


    




    
