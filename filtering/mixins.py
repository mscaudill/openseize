import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

class IIRViewer:
    """A mixin for IIR filters with methods for plotting the impulse
    response, frequency response and phase responses of an IIR filter."""

    def impulse_response(self):
        """Returns the impulse response of this IIR filter."""

        # 1-s array with unit pulse at 0th sample
        pulse = sps.unit_impulse(self.fs)
        
        if self.fmt == 'sos':
            resp = sps.sosfilt(self.coeffs, pulse)
        
        if self.fmt == 'ba':
            resp = sps.lfilter(*self.coeffs, pulse)
        
        return resp

    def frequency_response(self, scale='dB', worN=512, rope=-100):
        """Returns the frequency response of this IIR filter.

        Args:
            scale: str
                String in ('dB', 'abs', 'complex') that determines if
                returned response should be in decibels, magnitude, 
                or left as a complex number containing phase.
            worN: int
                The number of frequencies in [0, Nyquist) to evaluate
                response over. Default is 512.
            rope: float
                For plotting, all values below this region of pratical
                equivalence will be set to the rope value. Default is -100
                dB. Any filter response smaller than this will be set to
                -100 for plotting.

        Returns: array of frequencies (1 x worN) and an array of responses
        """

        if self.fmt == 'sos':
            freqs, h = sps.sosfreqz(self.coeffs, fs=self.fs, worN=worN)
        if self.fmt == 'ba':
            freqs, h = sps.freqz(*self.coeffs, fs=self.fs, worN=worN)

        if scale == 'dB':
            gain = 20 * np.log10(np.maximum(np.abs(h), 10**(rope/20)))
        elif scale == 'abs':
            gain = abs(h)
        elif scale == 'complex':
            gain = h

        return freqs, gain, scale

    def _plot_impulse(self, ax, **kwargs):
        """Plots this filters impulse response to a matplolib axis.

        Args:
            ax: matplotlib axis
                The axis instance where the plot will be displayed on.
            **kwargs: dict
                Any valid kwarg for matplotlib plot.
        """

        time = np.linspace(0, 1, self.fs)
        imp_response = self.impulse_response()
        ax.plot(time, imp_response, **kwargs)

    def _plot_response(self, ax, freqs, response, scale, alpha=0.2, **kwargs):
        """ """

        ax.plot(freqs, response, **kwargs)

        # get coordinates of pass and transition bands
        if self.btype == 'lowpass':
            pass_start = np.array([0])
            pass_width = self.fpass
            trans_start = self.fpass
            trans_width = self.fstop - trans_start

        elif self.btype == 'highpass':
            pass_start = self.fpass
            pass_width = np.atleast_1d(self.nyq) - pass_start
            trans_start = self.fstop
            trans_width = self.fpass - trans_start

        elif self.btype == 'bandpass':
            pass_start = np.atleast_1d(self.fpass[0])
            pass_width = np.atleast_1d(self.fpass[1] - pass_start)
            # two transition bands for bandpass
            trans_start = np.array([self.fstop[0], self.fpass[1]])
            trans_width = np.array([self.fpass[0] - self.fstop[0],
                                    self.fstop[1] - self.fpass[1]])

        elif self.btype == 'bandstop':
            # two passband starts and widths for bandstop
            pass_start = np.array([0, self.fstop[1]])
            pass_width = np.array([self.fstop, ax.get_xlim()[1]])

        bottom = ax.get_ylim()[0]
        top = 0 if scale=='dB' else 1
        height = top - bottom

        # construct passband rectangles and place on  axis
        pass_rects = [Rectangle((start, bottom), width, height,
                                 facecolor='tab:green', alpha=alpha) 
                      for start, width in zip(pass_start, pass_width)]
        [ax.add_patch(rect) for rect in pass_rects]

        # construct transition rectangles and place on axis
        trans_rects = [Rectangle((start, bottom), width, height,
            facecolor='red', alpha=alpha) for start,
                                 width in zip(trans_start, trans_width)]

        [ax.add_patch(rect) for rect in trans_rects]
       
        # construct a rectangle for the pass band attenuation
        attn_bottom = -self.gpass if scale=='dB' else 10**(-self.gpass/20)
        attn_top = self.gpass if scale=='dB' else 10**(self.gpass/20)
        attn_height = attn_top - attn_bottom
        attn_rects = [Rectangle((start, attn_bottom), width, attn_height,
                                 facecolor='None', edgecolor='gray',
                                 linestyle='dotted') 
                      for start, width in zip(pass_start, pass_width)]
        [ax.add_patch(rect) for rect in attn_rects]

        return ax

    def plot(self, size=(8,6), gridalpha=0.3, worN=2048):
        """ """

        fig, axarr = plt.subplots(3, 1, figsize=size)
        
        # Impulse response
        self._plot_impulse(axarr[0], color='tab:blue')
 
        # Frequency response in dB
        freqs, gain_dB, scale = self.frequency_response(scale='dB')
        self._plot_response(axarr[1], freqs, gain_dB, scale,
                color='tab:blue')

        ax2 = axarr[1].twinx()
        freqs, gainz, scale = self.frequency_response(scale='complex')
        angles = np.unwrap(np.angle(gainz, deg=True))
        ax2.plot(freqs, angles, color='k', alpha=0.2)
        
        # Frequency response in magnitude
        freqs, gain_abs, scale = self.frequency_response(scale='abs')
        self._plot_response(axarr[2], freqs, gain_abs, scale, color='tab:blue')
        
        [ax.grid(alpha=gridalpha) for ax in axarr]
        
        plt.ion()
        plt.tight_layout()
        plt.show()
