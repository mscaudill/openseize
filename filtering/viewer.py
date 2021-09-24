import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class FilterViewer:
    """Mixin for plotting impulse response, frequency responses"""

    @property
    def kind(self):
        """Returns the type name of this filter."""

        #search for filter type in inheritance tree
        names = [cls.__name__ for cls in type(self).__mro__]
        if 'FIR' in names:
            return 'FIR'
        elif 'IIR' in names:
            return 'IIR'
        else:
            raise TypeError('Filter must be of type FIR or IIR')
    
    def _impulse_response(self, nsamples):
        """Returns the impulse response of an IIR filter upto nsamples."""

        impulse = sps.unit_impulse(nsamples)
        if self.fmt == 'sos': 
            arr = sps.sosfilt(self.coeffs, impulse)
        elif self.fmt == 'ba':
            arr = sps.lfilter(*self.coeffs, impulse)
        elif self.fmt == 'zpk':
            sos = sps.zpk2sos(*self.coeffs)
            arr = sps.sosfilt(sos, impulse)
        return arr

    def _plot_impulse(self, ax, gridalpha, **kwargs):
        """Plots the impulse response of the filter to axis."""

        if self.kind == 'FIR':
            time = np.linspace(0, self.ntaps / self.fs, self.ntaps)
            ax.plot(time, self.coeffs, **kwargs)
        if self.kind == 'IIR':
            time = np.linspace(0, 1, self.fs)
            y = self._impulse_response(len(time))
            ax.plot(time, y, **kwargs)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(alpha=gridalpha)

    def _freqz(self, worN):
        """Returns the frequency response of this digital filter."""

        if self.kind == 'FIR':
            f, h = sps.freqz(self.coeffs, fs=self.fs, worN=worN)
        elif self.fmt == 'sos':
            f, h = sps.sosfreqz(self.coeffs, fs=self.fs, worN=worN)
        elif self.fmt == 'ba':
            f, h = sps.freqz(*self.coeffs, fs=self.fs, worN=worN)
        elif self.fmt == 'zpk':
            f, h = sps.freqz_zpk(*self.coeffs, fs=self.fs, worN=worN)
        return f, h

    def _plot_response(self, ax, freqs, response, transcolor, transalpha,
                       cutcolor, cutalpha, gridalpha, **kwargs):
        """Plots a frequency response to an axis.

        Args:
            freqs (arr):            array of frequencies for abscissa
            response (arr):         array of responses for ordinate
            transcolor (str):       color of the transition bands
            transalpha (float):     alpha of the transition bands
            cutcolor (str):         color of the cutoff frequencies
            cutalpha (float):       alpha of the cutoff frequencies
            gridalpha (float):      alpha of the background grid
            **kwargs:               passed to axis plot

        Returns: configured axis
        """

        ax.plot(freqs, response, **kwargs)
        #get left edges, top and bottom of transition bands
        edges = self.cutoff - 0.5 * self.width
        bottom, top = ax.get_ylim()
        height = top - bottom
        #create tranisition band rectanles
        rects = [Rectangle((e, bottom), self.width, height, 
                 facecolor=transcolor, alpha=transalpha) for e in edges]
        [ax.add_patch(rect) for rect in rects]
        #add transition cutoffs
        [ax.axvline(x=c, color=cutcolor, alpha=cutalpha) 
                for c in self.cutoff]
        ax.grid(alpha=gridalpha)
        return ax

    def _db_response(self, ax, transcolor, transalpha, cutcolor, cutalpha,
                     gridalpha, stopcolor, stopalpha, anglecolor,
                     anglealpha, worN, **kwargs):
        """Plots the gain of the filter in decibels & the phase response of
        the filter.

        Args:
            _freq_response args         see _freq_response method args
            stopcolor (str):            color str for min attenuation line
            stopalpha (float):          alpha value of the min attenuation
                                        line
            anglecolor (str):           color of phase response
            anglealpha (float):         alpha of phase response
            kwargs:                     valid kwargs for _freq_response

        Returns: a configured twin axis instance
        """

        #compute the frequency response
        freqs, h = self._freqz(worN)
        #convert the gain to dB and plot
        resp = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
        ax = self._plot_response(ax, freqs, resp, transcolor, transalpha,
                                 cutcolor, cutalpha, gridalpha, **kwargs)
        ax.set_ylabel('Gain (dB)', color='tab:blue')
        #add horizontal line for min stop attenuation
        ax.axhline(y=-self.stop_db, color=stopcolor, linestyle='--',
                   alpha=stopalpha)
        #create a twin axis for the phase response
        ax2 = ax.twinx()
        #obtain the angles and plot
        angles = np.unwrap(np.angle(h))
        ax2 = self._plot_response(ax2, freqs, angles, transcolor, 
                    transalpha, cutcolor, cutalpha, gridalpha, 
                    color=anglecolor, alpha=anglealpha, **kwargs)
        ax2.set_ylabel('Angle ($^\circ$)', color=anglecolor)
        return ax

    def _amp_response(self, ax, transcolor, transalpha, cutcolor, cutalpha,
                       gridalpha, ripplecolor, ripplealpha, worN, **kwargs):
        """Plots the gain of the filter & ripple constraint boundaries.

        Args:
            _freq_response args         see _freq_response method args
            ripplecolor (str):          color to apply to ripple boundaries
            ripplealpha (str):          alpha to apply to ripple boundaries
            kwargs:                     valid kwargs for _freq_response

        Returns: configured axis instance
        """

        freqs, h = self._freqz(worN)
        #compute the amplitude gain of the filter and plot
        resp = abs(h)
        ax = self._plot_response(ax, freqs, resp, transcolor, transalpha,
                                 cutcolor, cutalpha, gridalpha, **kwargs)
        #compute the ripple bounds of the passband & plot
        rbounds = [1 - self.pass_ripple, 1 + self.pass_ripple]
        [ax.axhline(y=rp, color=ripplecolor, linestyle='--',
                    alpha=ripplealpha) for rp in rbounds]
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain')
        return ax

    def view(self, figsize=(8,6), transcolor='pink', transalpha=0.25,
             cutcolor='r', cutalpha=0.3, gridalpha=0.15, stopcolor='gray',
             stopalpha=0.3, anglecolor='g', anglealpha=0.4, 
             ripplecolor='gray', ripplealpha=0.3, worN=1024, **kwargs):
        """Displays impulse, frequency, phase and gain reponse curves for
        a filter.

            figsize (tuple):        width, height tuple of figure
            transcolor (str):       color of the transition bands
            transalpha (float):     alpha of the transition bands
            cutcolor (str):         color of the cutoff frequencies
            cutalpha (float):       alpha of the cutoff frequencies
            gridalpha (float):      alpha of the background grid
            stopcolor (str):        color of min attenuation line
            stopalpha (float):      alpha of the min attenuation line
            anglecolor (str):       color of phase response
            anglealpha (float):     alpha of phase response
            ripplecolor (str):      color to apply to ripple boundaries
            ripplealpha (str):      alpha to apply to ripple boundaries
            worN (int):             number of frequencies to compute freq
                                    responses over (Default=1024)
            kwargs:                 kwargs passed to all axes subplot plot

        Returns: None
        """

        #create and set up three axes for plotting and config.
        fig, axarr = plt.subplots(3, 1, figsize=figsize)
        axarr[2].get_shared_x_axes().join(axarr[2], axarr[1])
        #call each subplot
        self._plot_impulse(axarr[0], gridalpha, **kwargs)
        self._db_response(axarr[1], transcolor, transalpha, cutcolor, 
                          cutalpha, gridalpha, stopcolor, stopalpha, 
                          anglecolor, anglealpha, worN, **kwargs)
        self._amp_response(axarr[2], transcolor, transalpha, cutcolor,
                            cutalpha, gridalpha, ripplecolor, ripplealpha,
                            worN, **kwargs)
        plt.tight_layout()
        plt.show()

