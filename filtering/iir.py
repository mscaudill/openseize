import abc
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from matplotlib.patches import Rectangle

from openseize.mixins import ViewInstance
from openseize.filtering.viewer import FilterViewer

class IIR(abc.ABC, ViewInstance, FilterViewer):
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
        fmt (str):                      format for coeffs of filter (Default
                                        is second order sections 'sos')
        -- Computed --
        order (int):                    number filter coeffs to achieve pass,
                                        stop and transition width criteria
        coeffs (arr):                   filter coeffs of the designed filter
                                        "h(n)"
        others:                         Each FIR may add additional computed
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
    def _build(self, fmt):
        """Builds and returns the coefficients of this IIR filter."""

    @abc.abstractmethod
    def _order(self):
        """Determines the lowest order filter to meet the pass &
        stop band attenuation criteria."""

    def apply(self, arr, axis=-1, phase_shift=True, **kwargs):
        """Apply this IIR to an ndarray of data along axis.

        Args:
            arr (ndarray):          array with sample along axis
            axis (int):             axis to filter along
            phase_shift (bool):     boolean to compensate for IIR phase
                                    shift (Default=True)
            kwargs:                 passed to sosfilt
        """

        #FIXME Handle large data with a generator using add-overlap meth
        #FIXME Handle when fmt is not sos
        if phase_shift:
            return sps.sosfiltfilt(self.sos, arr, axis=axis, **kwargs)
        else:
            return sps.sosfilt(self.sos, arr, axis=axis, **kwargs)


class DIIR(IIR):
    """A Digital Infinite Impulse Response filter represented in second-
    order sections (sos) format.

    IIR can create and apply filters of type; Butterworth, ChebyshevI,
    ChebyshevII, and Elliptical. The filter order is determined as
    the minimum order needed to meet the pass band gain 'gpass' while also
    meeting the minimum attenuation in the stop band 'gstop'.

    Attrs:
        cutoff (float or 1-D array):        freqs at which Gain of filter
                                            drops to <= gstop
        width (int):                        width of transition bewteen pass
                                            & stop bands
        fs (int):                           sampling rate in Hz
        btype (str):                        filter band type one of
                                            {lowpass, highpass, bandpass, 
                                            bandstop}
        ftype (str):                        a filter type must be one of
                                            {butter, cheby1, cheby2, ellip}
        gpass (float):                      maximum allowed attenuation in
                                            the pass band in dB (Default is 
                                            0.9 dB ~ 9% amplitude diff)
        gstop (float):                      minimum attenuation in the stop
                                            band (Default is 6dB ~ 50% amplitude
                                            diff)
        kwargs:                             passed to scipy iirfilter. These
                                            should include the pass and stop 
                                            ripples ('rp' & 'rs) for Chebyshev
                                            and Elliptical filter types

    Note: In this implementation all transition bands use the same width.
    Call iirfilter directly if asymmetric bands are needed.

    For details on filter design please see scipy's signal.iirfilter and
    signal.iirdesign functions.
    """

    def __init__(self, cutoff, width, fs, btype='lowpass', ftype='butter',
                 pass_ripple=0.005, stop_db=40, fmt='sos'):
        """Build a standard scipy IIR filter."""

        super().__init__(cutoff, width, fs, btype=btype,
                         pass_ripple=pass_ripple, stop_db=stop_db)
        self.ftype = ftype
        #add the pass and stop bands
        self._wp, self._ws = self._bands()
        self._gpass = -20 * np.log10(1-pass_ripple)
        self._gstop = stop_db
        #call build
        self._build()

    def _build(self):
        """Build this digital filter using the second order section fmt."""

        self.order = self._order()
        self.sos = sps.iirfilter(self.order, self.cutoff, btype=self.btype, 
                            ftype=self.ftype, output=self.fmt, fs=self.fs)

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

    


if __name__ == '__main__':

    f = DIIR([100, 200], width=30, fs=1000, btype='bandpass', ftype='butter',
            pass_ripple=0.005, stop_db=40)

    """
    time = 10
    fs = 5000
    nsamples = int(time * fs)
    t = np.linspace(0, time, nsamples)

    # make a small 10 Hz riding on top of a larger 100 Hz sinusoidal signal
    x = 0.5 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t) + \
            0.25*np.random.random(nsamples)
    y = 2 * np.sin(2 * np.pi * 10 * t) + 0.5 *np.random.random(nsamples) 

    arr = np.stack((x,y, 1.5*x, y))

    results = f.apply(arr, phase_shift=True)

    fig, axarr = plt.subplots(4,1)
    [axarr[idx].plot(row) for idx,row in enumerate(arr)]
    [axarr[idx].plot(row, color='r') for idx, row in enumerate(results)]
    plt.show()
    """
