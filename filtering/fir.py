import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import signaltools, firwin, freqz, convolve
from matplotlib.patches import Rectangle

from openseize.mixins import ViewInstance

class FIRViewer:
    """Mixin for plotting the impulse response, frequency response and delay
    of a FIR response filter."""

    def _impulse_response(self, ax, **kwargs):
        """Plots the impulse response of the FIR Filter."""

        #construct time array for impulse resp.
        time = np.linspace(0, self.ntaps / self.fs, self.ntaps)
        #make a legend showing tap count and plot
        msg = 'Impulse Response with {} taps'
        label = kwargs.pop('label', msg.format(self.ntaps))
        ax.plot(time, self.coeffs, label=label, **kwargs)
        ax.legend()
        
    def _magnitude_response(self, ax, **kwargs):
        """Plots the magnitude response of the FIR Filter."""

        #make a legend showing trans. width
        msg = 'Transition Bandwidth = {} Hz'
        label = kwargs.pop('label', msg.format(self.width))
        #compute and plot freq response "H(f)", f is in Hz
        f, h = freqz(self.coeffs, fs=self.fs)
        amplitude_ratio = 20 * np.log10(np.abs(h)) 
        ax.plot(f, amplitude_ratio, label=label, **kwargs)
        #plot transition band(s)
        color = kwargs.pop('facecolor', 'gray')
        alpha = kwargs.pop('alpha', 0.3)
        #get left edges, bottom, and height
        #l_edges = self.cutoff
        bottom, top = ax.get_ylim()
        height = top - bottom
        #build and add rectangles FIXME
        if self.btype == 'highpass':
            l_edges = self.cutoff - self.width
        elif self.btype == 'bandpass':
            l_edges = self.cutoff + [-self.width, 0]
        elif self.btype == 'lowpass':
            l_edges = self.cutoff
        elif self.btype == 'bandstop':
            l_edges = self.cutoff + [0, -self.width]


        rects = [Rectangle((le, bottom), self.width, height, 
                 facecolor=color, alpha=alpha) for le in l_edges]
        [ax.add_patch(rect) for rect in rects]
        ax.legend()
    
    def _delay(self, ax, **kwargs):
        """Plots the delay of the filter."""

        w, h = freqz(self.coeffs)
        #the group delay is the derivative of the phase of h wrt w
        delay = -np.diff(np.unwrap(np.angle(h))) / np.diff(w)
        ax.plot(w[1:]*self.fs*1/(2*np.pi), delay/self.fs, **kwargs)
    
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
        self._impulse_response(axarr[0], **pargs[0])
        self._magnitude_response(axarr[1], **pargs[1])
        self._delay(axarr[2], **pargs[2])
        plt.tight_layout()
        plt.show()


class FIR_I(ViewInstance, FIRViewer):
    """A type I Finitie Impulse Response filter constructed using the window
    method.

    Attrs:
        fs (int):                       sampling frequency in Hz
        nyq (int):                      nyquist frequency
        cutoff (float or 1-D array):    freqs at which Gain of the filter
                                        drops to <= -6dB
        width (int):                    width of transition bewteen pass and
                                        stop bands
        window (str):                   scipy signal window, must be one of
                                        (hanning, blackman, rectangular, 
                                        hamming)
        btype (str):                    type of filter must be one of
                                        {lowpass, highpass, bandpass, 
                                        bandstop}. Default is lowpass
        gpass (float):
        gstop (float):
        coeffs (arr):                   filter coeffs of the designed filter
                                        "h(n)"

    Scipy's firwin requires the number of taps to determine the transition 
    width between pass and stop bands. FIR_I uses the transition width and 
    calculates the needed taps to achieve the requested roll-off.

    Reference:
        1. Ifeachor E.C. and Jervis, B.W. (2002). Digital Signal Processing:
           A Practical Approach. Prentice Hall
    """

    def __init__(self, cutoff, width, fs, btype='lowpass', window='hamming',
                 gpass=0.05, gstop=40):
        """Initialize this FIR filter. """

        #0.05 dB ~ 0.5% amplitude variation
        # 40 dB ~ 99 % amplitude reduction

        self.fs = fs
        self.nyq = fs/2
        self.cutoff = np.atleast_1d(cutoff)
        self.norm_cutoff = self.cutoff / self.nyq
        self.width = width
        self.window = window
        self.btype, pass_zero = self._validate_btype(btype)
        self.gpass = gpass
        self.gstop = gstop
        #compute taps needed and call scipy firwin
        self.ntaps = self._tap_count()
        self.coeffs = firwin(self.ntaps, self.norm_cutoff, window=self.window,
                             pass_zero=pass_zero)

    def _validate_btype(self, btype):
        """Validates that the filter type is a valid filter type."""

        types = ['lowpass', 'highpass', 'bandpass', 'bandstop']
        if btype not in types:
            msg = ('filter of type {} is not a valid filter type. '
                    'Valid filter types are {}')
            raise ValueError(msg.format(btype, types))
        pass_zero = True if btype in ['lowpass', 'bandstop'] else False
        return btype, pass_zero

    def _tap_count(self):
        """ """

        w = self.width * len(self.cutoff)
        #normalize transition width 
        normed_fwidth = w / (2 * (self.fs))
        #convert gains from decibels to ratios
        gpass = 10 ** (-self.gpass / 20)
        gstop = 10 ** (-self.gstop / 20)
        ntaps = int(2/3 * np.log10(1/(10 * gpass * gstop)) / normed_fwidth)
        #FIXME understand
        ntaps *= len(self.cutoff)
        ntaps = ntaps + 1 if ntaps % 2 == 0 else ntaps
        return ntaps

    def apply(self, arr, phase_shift=True):
        """Apply this FIR to an ndarray of data.

        Args:
            arr (ndarray):      array with samples along last axis
            phase_shift (bool): whether to compensate for the filter's
                                phase shift (Default is True)
        
        Returns: ndarry of filtered signal values
        """

        mode = 'same' if phase_shift else 'full'
        return convolve(arr, self.coeffs[np.newaxis,:], mode=mode)

        
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

        
    fir = FIR_I([50, 70], width=5, fs=5000, btype='bandstop', gpass=0.05, gstop=40)
    fir.view()

    results = fir.apply(arr)

    fig, axarr = plt.subplots(4,1)
    [axarr[idx].plot(row) for idx,row in enumerate(arr)]
    [axarr[idx].plot(row, color='r') for idx, row in enumerate(results)]
    plt.show()
