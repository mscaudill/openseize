import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from matplotlib.patches import Rectangle

from openseize.mixins import ViewInstance

class IIRViewer:
    """Mixin for plotting the impulse, frequency and delay response of an
    IIR filter."""

    def _impulse_response(self, ax, **kwargs):
        """ """



    def _frequency_response(self, ax, **kwargs):
        """ """

        #make a legend showing trans. width
        msg = 'Transition Bandwidth = {} Hz'
        label = kwargs.pop('label', msg.format(self.width))
        
        #FIXME I'm the only line thats different
        #test if self has coeffs or sos attr
        f, h = sps.sosfreqz(self.sos, fs=self.fs)
        
        amplitude_ratio = 20 * np.log10(np.maximum(np.abs(h), 1e-5)) 
        ax.plot(f, amplitude_ratio, label=label, **kwargs)
        #plot transition band(s)
        color = kwargs.pop('facecolor', 'gray')
        alpha = kwargs.pop('alpha', 0.3)
        #get left edges, bottom, and height
        l_edges = self.cutoff - 0.5 * self.width
        bottom, top = ax.get_ylim()
        height = top - bottom
        #build and add rectangles
        rects = [Rectangle((le, bottom), self.width, height, 
                 facecolor=color, alpha=alpha) for le in l_edges]
        [ax.add_patch(rect) for rect in rects]
        ax.legend()

    def view(self, figsize=(8,6), **plt_kwargs):
        """Displays impulse and frequency response and delay of filter."""

        #create and set up three axes for plotting and config.
        fig, axarr = plt.subplots(3, 1, figsize=figsize)
        axarr[0].set_xlabel('Time (s)')
        axarr[0].set_ylabel('Amplitude')
        axarr[1].set_ylabel('Magnitude (dB)')
        axarr[2].set_xlabel('Frequency (Hz)')
        axarr[2].set_ylabel('Delay (s)')
        axarr[2].get_shared_x_axes().join(axarr[2], axarr[1])
        #obtain plot args for each axis dicts
        pargs = [dict(), dict(), dict()]
        for idx, dic in enumerate(plt_kwargs):
            pargs[idx].update(dic)
        #call each subplot
        #self._impulse_response(axarr[0], **pargs[0])
        self._frequency_response(axarr[1], **pargs[1])
        #self._delay(axarr[2], **pargs[2])
        plt.tight_layout()
        plt.show()



class IIR(ViewInstance, IIRViewer):
    """A digital infinite impulse response filter represented in second-
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
                 gpass=0.90, gstop=6.0, **kwargs):
        """Intialize filter by determing pass & stop bands & determine the
        required order to meet the pass and stop attenuation criteria. """

        #use cutoff and widths to get pass and stop bands
        self.btype, self.wp, self.ws = self._bands(cutoff, width, btype)
        self.cutoff = np.atleast_1d(cutoff)
        self.width = width
        self.fs = fs
        self.ftype = ftype
        self.gpass = gpass
        self.gstop = gstop
        self.sos = sps.iirfilter(self.order, self.cutoff, btype=btype, 
                            ftype=ftype, output='sos', fs=fs, **kwargs)

    def _bands(self, cutoff, width, btype):
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

        widths = {'lowpass': width, 'highpass': -width, 
                  'bandpass': np.array([-width, width]), 
                  'bandstop': np.array([width, -width])}
        cutoff = np.atleast_1d(cutoff)
        try:
            return btype, cutoff, cutoff + widths[btype]
        except KeyError:
            msg = ('band of type {} is not a valid band type. '
                    'Valid band types are {}')
            raise ValueError(msg.format(btype, widths.keys()))

    @property
    def order(self):
        """Determines the lowest order filter of ftype to meet the pass &
        stop band attenuation criteria."""

        forders = {'butter': sps.buttord, 'cheby1': sps.cheb1ord,
                   'cheby2': sps.cheb2ord, 'ellip': sps.ellipord}
        try:
            #fetch and call order func from scipy signal module
            return forders[self.ftype](self.wp, self.ws, self.gpass,
                                         self.gstop, fs=self.fs)[0]
        except KeyError:
            msg = ('filter of type {} is not a valid filter type. '
                    'Valid filter types are {}')
            raise ValueError(msg.format(self.ftype, forders.keys()))

    def apply(self, arr, axis=-1, phase_shift=True, **kwargs):
        """Apply this IIR to an ndarray of data along axis.

        Args:
            arr (ndarray):          array with sample along axis
            axis (int):             axis to filter along
            phase_shift (bool):     boolean to compensate for IIR phase
                                    shift (Default=True)
            kwargs:                 passed to sosfilt
        """

        if phase_shift:
            return sps.sosfiltfilt(self.sos, arr, axis=axis, **kwargs)
        else:
            return sps.sosfilt(self.sos, arr, axis=axis, **kwargs)



if __name__ == '__main__':

    f = IIR(100, width=10, fs=5000, btype='lowpass', ftype='butter',
            gpass=0.99, gstop=40)

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
