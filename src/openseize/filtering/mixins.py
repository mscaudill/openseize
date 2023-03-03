"""This module contains filter mixin classes endowing filters with the
abiility to plot their impulse and frequency responses to a Matplotlib
figure called the 'Viewer'.

For usage, please see opensieze.filtering.iir or fir modules
"""
import typing
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps
from matplotlib.patches import Rectangle


class Viewer:
    """A collection of common plotting methods for both IIR, FIR and 
    Parks-McClellan filters.

    All filters in openseize have the ability to plot their impulse response
    and frequency response to a matplotlib figure called the Viewer. This
    mixin is inherited by specific IIR, FIR and ParksMcClellan Viewers in
    this file. Each of these specific viewers is inherited by the 
    corresponding filter type (i.e. IIR, FIR, ParksMcClellan) in the 
    openseize filtering module.
    """

    def _plot_impulse(self, ax, **kwargs):
        """Plots the impulse response of this filter to a matplolib axis.

        Args:
            ax: matplotlib axis
                The axis to plot the response to.
            **kwargs: dict
                Any valid kwarg for matplotlib plot.
        """

        time = np.linspace(0, 1, self.fs)
        imp_response = self.impulse_response()
        ax.plot(time, imp_response, **kwargs)
        if hasattr(self, 'order'):
            N, wn = self.order
            text = 'Filter Order = {}'.format(N)
        elif hasattr(self, 'numtaps'):
            text = 'Num. Taps = {}'.format(self.numtaps)
        ax.text(.8, .8, text, transform=ax.transAxes, weight='bold')

        return ax

    def _plot_response(self, ax, freqs, response, **kwargs):
        """Plots the frequency response of this filter to a matplotlib axis.

        Args:
            ax: matplotlib axis instance
                The axis to plot the response to.
            freqs: 1-D array
                The frequencies at which the response was computed.
            response: 1-D array
                The response of this filter at each frequency.
            kwargs: dict
                Any valid kwarg to matplotlib plot for adjusting line
                properties.

        Returns: The input ax with plots added.
        """

        ax.plot(freqs, response, **kwargs)

        return ax

    def _plot_rectangles(self, ax, scale):
        """Plots pass, transition & attenuation rectangles for this filter
        to a matplotlib axis.

        Args:
            ax: matplotlib axis instance
                The axis where pass and tranisition bands will be plotted.
            scale: str
                The units of the gain response being plotted to ax. Must be
                one of 'dB' or 'abs' for decibels and relative magnitude
                respectively.

        Returns: The input ax with Rectangle patches added.
        """

        bands = np.stack((self.fpass, self.fstop), axis=0)
        trans_bands = np.stack((np.min(bands, axis=0), 
                                np.max(bands, axis=0))).T
       
        if self.btype == 'lowpass':
            pass_bands = np.array([[0, self.fpass[0]]])
        
        elif self.btype == 'highpass':
            pass_bands = np.array([[self.fpass[0], self.nyq]])

        elif self.btype == 'bandpass':
            pass_bands = np.atleast_2d(self.fpass)

        elif self.btype == 'bandstop':
            pass_bands = np.array([[0, self.fpass[0]], 
                                  [self.fpass[1], self.nyq]])

        else: # Multiband case
            pass_bands = self.bands[np.where(self.desired)[0]]
            trans_bands = np.stack((self.bands[:-1,1], 
                                    self.bands[1:, 0]), axis=1)
        
        # Get bottom and top of pass and trans bands
        b = ax.get_ylim()[0]
        top = 0 if scale=='dB' else 1
        h = top - b
    
        # Plot pass band Rectangles from coords [(left, b), width, h]
        pass_coords = [[(p[0], b), p[1] - p[0], h] for p in pass_bands]
        pass_rects = [Rectangle(*pass_coord, fc='tab:green', alpha=0.2) 
                      for pass_coord in pass_coords]
        [ax.add_patch(rect) for rect in pass_rects]

        # Plot transition Rectangles from coords [(left, b), width, h]
        trans_coords = [[(t[0], b), t[1] - t[0], h] for t in trans_bands]
        trans_rects = [Rectangle(*trans_coord, fc='red', alpha=0.2) 
                       for trans_coord in trans_coords]
        [ax.add_patch(rect) for rect in trans_rects]

        # Get bottom and top of each pass band attenuation Rectangle
        att_b = -self.gpass if scale=='dB' else 10 ** (-self.gpass / 20)
        att_top = self.gpass if scale=='dB' else 10 ** (self.gpass / 20)
        att_h = att_top - att_b

        # Plot attenuation Rectangles from coords [(left, b), width, h]
        a_coords = [[(p[0], att_b), p[1] - p[0], att_h] for p in pass_bands]
        att_rects = [Rectangle(*a_coord, fc='None', edgecolor='gray',
                                linestyle='dotted') for a_coord in a_coords]
        [ax.add_patch(rect) for rect in att_rects]

        return ax
    
    @typing.no_type_check #mypy missing frequency_response attr.
    def plot(self, 
             size: Tuple[int, int] = (8,6), 
             gridalpha: float = 0.3, 
             worN: int = 2048, 
             rope: float = -100,
             axarr: Optional[Sequence[plt.Axes]] = None, 
             show: bool = True
    ) -> Optional[Sequence[plt.Axes]]:
        """Plots the impulse and frequency response of this filter.

        Args:
            size: tuple
                The figure size to display for the plots. Default is 8 x 6.
            gridalpha: float in [0, 1]
                The alpha transparency of each subplots grid. Default is 0.3
            worN: int
                The number of frequencies to compute the gain and phase
                responses over. Default is 2048 frequencies.
            rope: float
                For plotting, all values below this region of practical
                equivalence will be set to this value. Default is -100
                dB. Any filter response smaller than this will be set to
                -100 for plotting.
            axarr: A Matplotlib axis array.
                An optional axis array to plot the impulse and frequency
                responses to. Default None means a new axis is created.
        """

        if axarr is None:
            fig, axarr = plt.subplots(3, 1, figsize=size)
        axarr[1].get_shared_x_axes().join(axarr[1], axarr[2])
        
        # Plot impulse response and configure axis
        self._plot_impulse(axarr[0], color='tab:blue')
        axarr[0].set_xlabel('Time (s)', fontweight='bold')
        axarr[0].set_ylabel('Amplitude (au)', color='k', weight='bold')
        axarr[0].spines['right'].set_visible(False)
        axarr[0].spines['top'].set_visible(False)
 
        # Plot frequency response in dB and configure axis
        freqs, g_dB, scale = self.frequency_response('dB', worN, rope)
        color = 'tab:blue'
        self._plot_response(axarr[1], freqs, g_dB, color=color)
        axarr[1].set_ylabel('Gain (dB)', color=color, weight='bold')
        axarr[1].xaxis.set_ticklabels([])
        axarr[1].spines['top'].set_visible(False)
        # add pass and transition rectangles
        self._plot_rectangles(axarr[1], scale)

        # Plot phase response of this filter to twin axis
        ax2 = axarr[1].twinx()
        freqs, g_z, scale = self.frequency_response('complex', worN, rope)
        angles = np.unwrap(np.angle(g_z, deg=False))
        color = 'tab:orange'
        ax2.plot(freqs, angles, color=color, alpha=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.set_ylabel('Angle (radians)', color=color, weight='bold')
        axarr[2].spines['right'].set_visible(False)
        axarr[2].spines['top'].set_visible(False)

        # Plot frequency response in relative magnitude and configure axis
        freqs, g_abs, scale = self.frequency_response('abs', worN, rope)
        color = 'tab:blue'
        self._plot_response(axarr[2], freqs, g_abs, color=color)
        axarr[2].set_xlabel('Frequency (Hz)', weight='bold')
        axarr[2].set_ylabel('Gain (au)', color='k', weight='bold')
        # add pass and transition rectangles
        self._plot_rectangles(axarr[2], scale)

        # Configure axes grids
        [ax.grid(alpha=gridalpha) for ax in axarr]
        plt.tight_layout()
    
        if show:
            plt.show() 
        else:
            return axarr
        

class IIRViewer(Viewer):
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

    def frequency_response(self, scale, worN, rope):
        """Returns the frequency response of this IIR filter.

        Args:
            scale: str
                String in ('dB', 'abs', 'complex') that determines if
                returned response should be in decibels, magnitude, 
                or left as a complex number containing phase.
            worN: int
                The number of frequencies in [0, Nyquist) to evaluate
                response over.
            rope: float
                For plotting, all values below this region of practical
                equivalence will be set to this value. E.g if rope = -100
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


class FIRViewer(Viewer):
    """A mixin for FIR filters with methods for plotting the impulse
    response, frequency response and phase response of an FIR filter."""

    def impulse_response(self):
        """Returns the impulse response of this FIR filter."""

        # 1-s array with unit pulse at 0th sample
        pulse = sps.unit_impulse(self.fs)
        resp = np.convolve(self.coeffs, pulse, mode='full')
        resp = resp[0:len(pulse)]
        
        return resp

    def frequency_response(self, scale, worN, rope):
        """Returns the frequency response of this IIR filter.

        Args:
            scale: str
                String in ('dB', 'abs', 'complex') that determines if
                returned response should be in decibels, magnitude, 
                or left as a complex number containing phase.
            worN: int
                The number of frequencies in [0, Nyquist) to evaluate
                response over.
            rope: float
                For plotting, all values below this region of practical
                equivalence will be set to this value. E.g if rope = -100
                dB. Any filter response smaller than this will be set to
                -100 for plotting.

        Returns: array of frequencies (1 x worN) and an array of responses
        """

        freqs, h = sps.freqz(self.coeffs, fs=self.fs, worN=worN)

        if scale == 'dB':
            gain = 20 * np.log10(np.maximum(np.abs(h), 10**(rope/20)))
        elif scale == 'abs':
            gain = abs(h)
        elif scale == 'complex':
            gain = h

        return freqs, gain, scale


