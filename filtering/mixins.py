import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

class IIRDesign:
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
                equivalence will be set to this value. Default is -100
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
        order, wn = self.order
        ax.text(.70, .6, 'Filter Order = {}'.format(order), 
                transform=ax.transAxes, fontweight='bold', color='tab:blue')

    def _band_coords(self, ax, scale):
        """Returns a list of coordinates for each pass and transition band
        rectangle for plotting.

        Args:
            ax: matplotlib axis instance
                The axis where pass and tranisition bands will be plotted.
            scale: str
                The units of the gain response being plotted to ax. Must be
                one of 'dB' or 'abs' for decibels and relative magnitude
                respectively.

        Returns: a list of tuple sequence arguments for matplotlib's
                 Rectangle patch one per band to be plotted.
        """

        if self.btype == 'lowpass':
            pass_bands = np.array([[0, self.fpass[0]]])
            trans_bands = np.array([[self.fpass[0], self.fstop[0]]])
        
        elif self.btype == 'highpass':
            pass_bands = np.array([[self.fpass[0], self.nyq]])
            trans_bands = np.array([[self.fstop[0], self.fpass[0]]])

        elif self.btype == 'bandpass':
            pass_bands = np.atleast_2d(self.fpass)
            #bandpass has two transition bands
            trans_bands = np.array([[self.fstop[0], self.fpass[0]],
                                   [self.fpass[1], self.fstop[1]]])

        elif self.btype == 'bandstop':
            #bandstop has two pass and transition bands
            pass_bands = np.array([[0, self.fpass[0]], 
                                  [self.fpass[1], self.nyq]])
            trans_bands = np.array([[self.fpass[0], self.fstop[0]],
                                   [self.fstop[1], self.fpass[1]]]) 

        # Get bottom and top of pass and trans bands
        bottom = ax.get_ylim()[0]
        top = 0 if scale=='dB' else 1
        height = top - bottom
    
        # returned coords are (left, bottom), width, height for each band
        pc = [[(p[0], bottom), p[1] - p[0], height] for p in pass_bands]
        tc = [[(t[0], bottom), t[1] - t[0], height] for t in trans_bands]

        # get the pass attenuation coords
        abottom = -self.gpass if scale=='dB' else 10**(-self.gpass/20)
        atop = self.gpass if scale=='dB' else 10**(self.gpass/20)
        aheight = atop - abottom

        # returned coords are (left, bottom), width, height for each band
        ac = [[(p[0], abottom), p[1] - p[0], aheight] for p in pass_bands]

        return pc, tc, ac

    def _plot_response(self, ax, freqs, response, scale, **kwargs):
        """Plots the frequency response of this filter and the pass and
        transition bands.

        Args:
            ax: matplotlib axis instance
                The axis to plot the response to.
            freqs: 1-D array
                The frequencies at which the response was computed.
            response: 1-D array
                The response of this filter at each frequency.
            scale: str
                The units of the response. Should be one of 'dB' or 'abs'
                for decibels and relative magnitudes respectively.
            kwargs: dict
                Any valid kwarg to matplotlib plot for adjusting line
                properties. The rectangular band properties are not
                configurable.

        Returns: The input ax with plots and rectangles added.
        """

        ax.plot(freqs, response, **kwargs)

        pass_coords, trans_coords, att_coords = self._band_coords(ax, scale)
        
        # construct passband rectangles and place on  axis
        pass_rects = [Rectangle(*pass_coord, fc='tab:green', alpha=0.2) 
                      for pass_coord in pass_coords]
        [ax.add_patch(rect) for rect in pass_rects]

        # construct transition rectangles and place on axis
        trans_rects = [Rectangle(*trans_coord, fc='red', alpha=0.2) 
                       for trans_coord in trans_coords]
        [ax.add_patch(rect) for rect in trans_rects]

        # construct attenuation rectangle in pass bands
        attn_rects = [Rectangle(*att_coord, fc='None', edgecolor='gray',
                                linestyle='dotted') 
                      for att_coord in att_coords]
        [ax.add_patch(rect) for rect in attn_rects]

        return ax

    def plot(self, size=(8,6), gridalpha=0.3, worN=2048):
        """Plots the impulse and frequency response of this IIR filter.

        Args:
            size: tuple
                The figure size to display for the plots. Default is 8 x 6.
            gridalpha: float in [0, 1]
                The alpha transparency of each subplots grid. Default is 0.3
            worN: int
                The number of frequencies to compute the gain and phase
                responses over. Default is 2048 frequencies.
        """

        fig, axarr = plt.subplots(3, 1, figsize=size)
        axarr[1].get_shared_x_axes().join(axarr[1], axarr[2])
        
        # Plot impulse response and configure axis
        self._plot_impulse(axarr[0], color='tab:blue')
        axarr[0].set_xlabel('Time (s)', fontweight='bold')
        axarr[0].set_ylabel('Amplitude (au)', color='k', weight='bold')
        axarr[0].spines['right'].set_visible(False)
        axarr[0].spines['top'].set_visible(False)

 
        # Plot frequency response in dB and configure axis
        freqs, g_dB, scale = self.frequency_response(scale='dB')
        color = 'tab:blue'
        self._plot_response(axarr[1], freqs, g_dB, scale, color=color)
        axarr[1].set_ylabel('Gain (dB)', color=color, weight='bold')
        axarr[1].xaxis.set_ticklabels([])
        axarr[1].spines['top'].set_visible(False)

        # Plot phase response of this filter to twin axis
        ax2 = axarr[1].twinx()
        freqs, g_z, scale = self.frequency_response(scale='complex')
        angles = np.unwrap(np.angle(g_z, deg=True))
        color = 'tab:orange'
        ax2.plot(freqs, angles, color=color, alpha=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.set_ylabel('Phase ($^\circ$)', color=color, weight='bold')
        axarr[2].spines['right'].set_visible(False)
        axarr[2].spines['top'].set_visible(False)
        
        # Plot frequency response in relative magnitude and configure axis
        freqs, g_abs, scale = self.frequency_response(scale='abs')
        color = 'tab:blue'
        self._plot_response(axarr[2], freqs, g_abs, scale, color=color)
        axarr[2].set_xlabel('Frequency (Hz)', weight='bold')
        axarr[2].set_ylabel('Gain (au)', color='k', weight='bold')

        # Configure axes grids
        [ax.grid(alpha=gridalpha) for ax in axarr]
        
        plt.tight_layout()
        plt.show()
