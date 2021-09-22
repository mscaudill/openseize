import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import signaltools, firwin, freqz, convolve, kaiserord
from scipy.signal.windows import kaiser
from matplotlib.patches import Rectangle

from openseize.mixins import ViewInstance
from openseize.filtering.viewer import FilterViewer

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

        #TODO
        #compute and plot freq response "H(f)", f is in Hz
        f, h = freqz(self.coeffs, fs=self.fs, worN=2000)
        amplitude_ratio = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
        gain = abs(h)
        ax.plot(f, amplitude_ratio, label=label, **kwargs)
        #plot transition band(s)
        color = kwargs.pop('facecolor', 'pink')
        alpha = kwargs.pop('alpha', 0.25)
        #get left edges, bottom, and height
        bottom, top = ax.get_ylim()
        height = top - bottom
        l_edges = self.cutoff - 0.5 * self.width

        rects = [Rectangle((le, bottom), self.width, height, 
                 facecolor=color, alpha=alpha) for le in l_edges]
        [ax.add_patch(rect) for rect in rects]
        [ax.axvline(x=cut, color='r', alpha=0.3) for cut in self.cutoff]
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


class FIR_I(ViewInstance, FilterViewer):
    """A type I Finitie Impulse Response filter constructed using the Kaiser
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
                                        band as a percentage(Default= 0.5%)
        stop_attenuation (float):       minimum attenuation to achieve at
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
        2. Ballenger (2000). Digital Processing of signals: Theory and 
           Practice 3rd ed. Wiley
        3. Oppenheim, Schafer, "Discrete-Time Signal Processing", pp.475-476.
    """

    def __init__(self, cutoff, width, fs, btype='lowpass', pass_ripple=0.5, 
                 stop_attenuation=40):
        """Initialize this FIR filter."""

        self.fs = fs
        self.nyq = fs/2
        self.cutoff = np.atleast_1d(cutoff)
        self.norm_cutoff = self.cutoff / self.nyq
        self.width = width
        self.btype = btype
        self.window = 'kaiser'
        self.pass_ripple = pass_ripple / 100
        self.stop_attenuation = stop_attenuation
        #compute taps needed and call scipy firwin
        self.ntaps, self.beta = self._tap_count()
        self.coeffs = firwin(self.ntaps, self.norm_cutoff, 
                             window=('kaiser', self.beta), 
                             pass_zero=self.btype, scale=False)

    @property
    def ftype(self):
        """Returns the filter type."""

        return 'fir'

    def _tap_count(self):
        """Returns the minimum number of taps needed for this FIR's
        attenuation and transition width criteria with a Kaiser window.

        Oppenheim, Schafer, "Discrete-Time Signal Processing", pp.475-476.
        """

        #find most restrictive dB criteria
        pass_db = -20 * np.log10(self.pass_ripple)
        design_param = max(pass_db, self.stop_attenuation)
        #compute taps and shape parameter
        ntaps, beta = kaiserord(design_param, self.width/(self.nyq))
        #Symmetric FIR type I requires odd tap num
        ntaps = ntaps + 1 if ntaps % 2 == 0 else ntaps
        return ntaps, beta

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

        
    fir = FIR_I([100, 200], width=30, fs=1000, btype='bandpass', 
                pass_ripple=.5, stop_attenuation=54)
    fir.view()

    """
    results = fir.apply(arr)

    fig, axarr = plt.subplots(4,1)
    [axarr[idx].plot(row) for idx,row in enumerate(arr)]
    [axarr[idx].plot(row, color='r') for idx, row in enumerate(results)]
    plt.show()
    """
