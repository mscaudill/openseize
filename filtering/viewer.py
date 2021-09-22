import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class FilterViewer:
    """Mixin for plotting impulse response, frequency response, & the phase
    response of an openseize filter."""

    def _impulse_response(self, ax):
        """Plots the impulse response of the filter to an axis."""

        if self.ftype == 'fir':
            time = np.linspace(0, self.ntaps / self.fs, self.ntaps)
            ax.plot(time, self.coeffs)
            ax.set_xlabel('time (s)')
            ax.set_ylabel('Amplitude')

    def _freq_response(self, ax, f, resp):
        """ """

        ax.plot(f, resp)
        #plot transition band(s)
        color = 'pink'
        alpha = 0.25
        #get left edges, bottom, and height
        l_edges = self.cutoff - 0.5 * self.width
        bottom, top = ax.get_ylim()
        height = top - bottom
        #add rectangles and cutoff lines
        rects = [Rectangle((le, bottom), self.width, height, 
                 facecolor=color, alpha=alpha) for le in l_edges]
        [ax.add_patch(rect) for rect in rects]
        [ax.axvline(x=cut, color='r', alpha=0.3) for cut in self.cutoff]
        return ax

    def _db_response(self, ax):
        """Plots the gain of the filter in decibels."""

        if self.ftype == 'fir':
            f, h = sps.freqz(self.coeffs, fs=self.fs, worN=2000)
        elif self.ftype == 'iir':
            f, h = sps.sosfreqz(self.sos, fs=self.fs)
        resp = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
        ax = self._freq_response(ax, f, resp)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        return ax

    def _gain_response(self, ax):
        """ """

        if self.ftype == 'fir':
            f, h = sps.freqz(self.coeffs, fs=self.fs, worN=2000)
        elif self.ftype == 'iir':
            f, h = sps.sosfreqz(self.sos, fs=self.fs)
        resp = abs(h)
        ax = self._freq_response(ax, f, resp)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain')
        return ax

    def _phase(self, ax):
        pass

    def view(self, figsize=(8,6)):
        """Displays impulse and frequency response and delay of filter."""

        #create and set up three axes for plotting and config.
        fig, axarr = plt.subplots(3, 1, figsize=figsize)
        axarr[2].get_shared_x_axes().join(axarr[2], axarr[1])
        #call each subplot
        self._impulse_response(axarr[0])
        self._db_response(axarr[1])
        self._gain_response(axarr[2])
        plt.tight_layout()
        plt.show()
