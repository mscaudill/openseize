import abc
import numpy as np
import scipy.signal as sps
from functools import partial

from openseize.core import mixins
from openseize.core.producer import producer
from openseize.core import numerical as onum
from openseize.filtering.viewer import FilterViewer

class IIRBase(abc.ABC, mixins.ViewInstance, FilterViewer):
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

    def apply(self, x, chunksize, axis, phase_correct=True, **kwargs):
        """Apply this IIR to an ndarray of data along axis.

        Args:
            x (producer or array-like):          

        """
        #FIXME DOCS

        pro = producer(x, chunksize, axis, **kwargs)
        #build a producer of filtered values
        if not phase_correct:
            result = onum.sosfilt(pro, self.coeffs, chunksize, axis,
                                  **kwargs)
        else:
            result = onum.sosfiltfilt(pro, self.coeffs, chunksize, axis, 
                                      **kwargs)
        #if input is array -> return an array
        if isinstance(x, np.ndarray):
            result = np.concatenate([arr for arr in res], axis=axis)
        return result


class IIR(IIRBase):
    """A Digital Infinite Impulse Response filter.

    IIR can create and apply filters of type; Butterworth, ChebyshevI,
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
        ftype (str):                    a filter type must be one of:
                                        'butter', 'cheby1', 'cheby2' or
                                        'ellip' (Default is 'butter')
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

        
        #FIXME should pass_ripple just be in dB to be consistent?
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

    from openseize.io.readers import EDF
    import matplotlib.pyplot as plt
    import time
   
    iir = IIR(500, width=100, fs=5000, btype='lowpass', ftype='butter',
            pass_ripple=0.005, stop_db=40, fmt='sos')

    PATH = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    edf = EDF(PATH)
    pro = producer(edf, chunksize=30e6, axis=-1)
    result = iir.apply(pro, chunksize=30e6, axis=-1, phase_correct=True)

    t0 = time.perf_counter()
    fig, axarr = plt.subplots(4,1)
    fig2, axarr2 = plt.subplots(4,1)
    for idx, arr in enumerate(result):
        if idx == 0: 
            [axarr[idx].plot(row[0:50000], color='r') for idx, row in 
             enumerate(arr)]
        elif idx == 1: 
            [axarr2[idx].plot(row[0:50000], color='r') for idx, row in 
             enumerate(arr)]
        else:
            break

    print('elapsed {} s'.format(time.perf_counter() - t0))

    for idx, arr in enumerate(pro):
        if idx == 0:
            [axarr[idx].plot(row[0:50000], color='b', alpha=0.5) for idx, row in 
                    enumerate(arr)]
        elif idx == 1: 
            [axarr2[idx].plot(row[0:50000], color='b', alpha=0.5) for idx, row in 
             enumerate(arr)]

        else:
            break

    plt.show()


    """
    for idx, filt in enumerate(result):
        if idx == 0:
            fig, axarr = plt.subplots(4,1)
            [axarr[idx].plot(row[0:50000], color='r') for idx, row in 
             enumerate(filt)]
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
    """




    """
    t0 = time.perf_counter()
    for idx, arr in enumerate(result):
        print('Filtered array # {}'.format(idx))
    print('Filtering completed in {} s'.format(time.perf_counter() - t0))
    """

    """
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

    pro = iir.apply(arr, chunksize=10000, axis=-1, phase=False)
    pro2 = iir.apply(arr, chunksize=10000, axis=-1, phase=True)
    results = np.concatenate([arr for arr in pro], axis=-1)
    results2 = np.concatenate([arr for arr in pro2], axis=-1)

    spresults = sps.sosfiltfilt(iir.coeffs, arr, axis=-1, padtype=None)
    #spresults = sps.sosfiltfilt(iir.coeffs, arr, axis=-1, padtype='odd')

    #print(np.allclose(results2, spresults))
    
    plt.ion()
    fig, axarr = plt.subplots(2,1)
    [axarr[idx].plot(row) for idx,row in enumerate(arr)]
    [axarr[idx].plot(row, color='g', label='forward') 
                     for idx, row in enumerate(results)]
    [axarr[idx].plot(row, color='orange', label='scipy') 
                     for idx, row in enumerate(spresults)]
    [axarr[idx].plot(row, color='r', linestyle='--', label='ops') 
            for idx, row in enumerate(results2)]
    axarr[0].legend()
    """


    """
    fig1, axarr1 = plt.subplots(2,1)
    [axarr1[idx].plot(row) for idx,row in enumerate(arr)]
    [axarr1[idx].plot(row, color='r') for idx, row in enumerate(spresults)]
    """

