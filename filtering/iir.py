import abc
import numpy as np
import scipy.signal as sps

from openseize.types import mixins
from openseize.types.producer import producer
from openseize.filtering.viewer import FilterViewer
from openseize.tools import numerical

class IIR(abc.ABC, mixins.ViewInstance, FilterViewer):
    """Abstract Infinite Impulse Response Filter defining required and common
    methods used by all IIR subclasses.

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
        fmt (str):                      representation format for coeffs 
                                        of filter (Default is second-order 
                                        sections 'sos')
        -- Computed --
        order (int):                    number filter coeffs to achieve pass,
                                        stop and transition width criteria
        coeffs (arr):                   coeffs of IIR transfer function
        others:                         Each IIR may add additional computed
                                        attrs
    """

    def __init__(self, cutoff, width, fs, btype='lowpass', pass_ripple=0.005,
                 stop_db=40, fmt='sos'):
        """Initializes & builds this IIR filter."""

        self.fs = fs
        self.nyq = fs/2
        self.cutoff = np.atleast_1d(cutoff)
        self.norm_cutoff = self.cutoff / self.nyq
        self.width = width
        self.btype = btype
        self.pass_ripple = pass_ripple
        self.stop_db = stop_db
        self.fmt = fmt

    @abc.abstractmethod
    def _order(self):
        """Determines the lowest order filter to meet the pass &
        stop band attenuation criteria."""

    @abc.abstractmethod
    def _build(self, fmt):
        """Builds and returns the coefficients of this IIR filter."""

    def apply(self, x, chunksize, axis=-1, phase=True, **kwargs):
        """Apply this IIR to an ndarray of data along axis.

        Args:
            x (producer or array-like):          
        """

        gen = numerical.batch_sosfilt(self.coeffs, x, chunksize, axis=axis)
        return producer(gen, chunksize=chunksize, axis=axis)


class DIIR(IIR):
    """A Digital Infinite Impulse Response filter.

    DIIR can create and apply filters of type; Butterworth, ChebyshevI,
    ChebyshevII, and Elliptical. The filter order is determined as
    the minimum order needed to meet the pass band ripple while also
    meeting the minimum attenuation in the stop band.

    Attrs:
        fs (int):                       sampling frequency in Hz
        nyq (int):                      nyquist frequency
        cutoff (float or 1-D array):    freqs at which Gain of the filter
                                        drops to <= -6dB
        norm_cutoff:                    cutoff normed to nyquist
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
        fmt (str):                      representation format for coeffs 
                                        of filter (Default is second-order 
                                        sections 'sos')
        -- Computed --
        order (int):                    number filter coeffs to achieve pass,
                                        stop and transition width criteria
        coeffs (arr):                   coeffs of IIR transfer function

    Note: In this implementation all transition bands use the same width.
    Call iirfilter directly if asymmetric bands are needed.

    For details on filter design please see scipy's signal.iirfilter and
    signal.iirdesign functions.
    """

    def __init__(self, cutoff, width, fs, btype='lowpass', ftype='butter',
                 pass_ripple=0.005, stop_db=40, fmt='sos'):
        """Build a standard scipy IIR filter."""

        super().__init__(cutoff, width, fs, btype=btype,
                         pass_ripple=pass_ripple, stop_db=stop_db, fmt=fmt)
        #store the scipy filter type
        self.ftype = ftype
        #add the pass and stop bands
        self._wp, self._ws = self._bands()
        #compute the max ripple and min attenuation in dB
        self._gpass = -20 * np.log10(1-pass_ripple)
        self._gstop = stop_db
        #compute the order needed to meet width, pass & stop criteria
        self.order = self._order()
        #call build
        self.coeffs = self._build()

    def _bands(self):
        """Returns the pass and stop bands for a filter of btype and
        transition width.

        Args:
            cutoff (float or 1-D array):        freqs at which filter gain
                                                drops to gstop decibels.
            width (int):                        width of transition between
                                                pass and stop bands
            btype (str):                        type of filter must be one 
                                                of {lowpass, highpass, 
                                                bandpass, bandstop}. Default
                                                is lowpass.
        Returns: filter order (int) and the Wn the critical freq at which
                 filter gain reaches gstop
        """

        w = self.width
        widths = {'lowpass': w/2, 'highpass': -w/2, 
                  'bandpass': np.array([-w, w])/2, 
                  'bandstop': np.array([w, -w])/2}
        try:
            return self.cutoff, self.cutoff + widths[self.btype]
        except KeyError:
            msg = ('band of type {} is not a valid band type. '
                    'Valid band types are {}')
            raise ValueError(msg.format(btype, widths.keys()))

    def _order(self):
        """Determines the lowest order filter of ftype to meet the pass &
        stop band attenuation criteria."""

        forders = {'butter': sps.buttord, 'cheby1': sps.cheb1ord,
                   'cheby2': sps.cheb2ord, 'ellip': sps.ellipord}
        try:
            #fetch and call order func from scipy signal module
            return forders[self.ftype](self._wp, self._ws, self._gpass,
                                         self._gstop, fs=self.fs)[0]
        except KeyError:
            msg = ('filter of type {} is not a valid filter type. '
                    'Valid filter types are {}')
            raise ValueError(msg.format(self.ftype, forders.keys()))

    def _build(self):
        """Build this digital filter using this filter's fmt."""

        return sps.iirfilter(self.order, self.cutoff, rp=self._gpass,
                    rs=self.stop_db, btype=self.btype, ftype=self.ftype, 
                    output=self.fmt, fs=self.fs)



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    iir = DIIR(50, width=10, fs=5000, btype='lowpass', ftype='butter',
            pass_ripple=0.005, stop_db=40, fmt='sos')

    #iir.view()
    time = 10
    fs = 5000
    nsamples = int(time * fs)
    t = np.linspace(0, time, nsamples)

    # make a small 10 Hz riding on top of a larger 100 Hz sinusoidal signal
    x = 0.5 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t) + \
            0.25*np.random.random(nsamples)
    y = 2 * np.sin(2 * np.pi * 22 * t) + 0.5 *np.random.random(nsamples) 

    arr = np.stack((x, y))

    pro = iir.apply(arr, chunksize=10000, axis=-1)
    results = np.concatenate([arr for arr in pro], axis=-1)

    spresults = sps.sosfilt(iir.coeffs, arr, axis=-1)

    print(np.allclose(results, spresults))

    plt.ion()
    fig, axarr = plt.subplots(2,1)
    [axarr[idx].plot(row) for idx,row in enumerate(arr)]
    [axarr[idx].plot(row, color='r') for idx, row in enumerate(results)]

    """
    fig1, axarr1 = plt.subplots(2,1)
    [axarr1[idx].plot(row) for idx,row in enumerate(arr)]
    [axarr1[idx].plot(row, color='r') for idx, row in enumerate(spresults)]
    """

