import abc
import numpy as np
import scipy.signal as sps

from openseize.core import mixins
from openseize.core.producer import producer
from openseize.core import numerical as nm

# TODO DEPRECATE once viewer is in base
from openseize.filtering.viewer import FilterViewer
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class IIRDesign:
    """ """

    def impulse_response(self):
        """Returns the impulse response of this IIR filter."""

        # 1-s array with unit pulse at 0th sample
        pulse = sps.unit_impulse(self.fs)
        
        if self.fmt == 'sos':
            resp = sps.sosfilt(self.coeffs, pulse)
        
        if self.fmt == 'ba':
            resp = sps.lfilter(*self.coeffs, pulse)
        
        return resp

    def frequency_response(self, scale='dB', worN=512):
        """Returns the frequency response of this IIR filter.

        Args:
            scale: str
                String in ('dB', 'abs', 'complex') that determines if
                returned response should be in decibels, magnitude, 
                or left as a complex number containing phase.
            worN: int
                The number of frequencies in [0, Nyquist) to evaluate
                response over. Default is 512.

        Returns: array of frequencies (1 x worN) and an array of responses
        """

        if self.fmt == 'sos':
            freqs, h = sps.freqz(self.coeffs, fs=self.fs, worN=worN)
        if self.fmt == 'ba':
            freqs, h = sps.freqz(*self.coeffs, fs=self.fs, worN=worN)

        if scale == 'dB':
            gain = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
        elif scale == 'abs':
            gain = abs(h)
        elif scale == 'complex':
            gain = h

        return freqs, gain

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

    def _plot_response(self, ax, freqs, response, cutcolor, trans_alpha,
                       stopcolor, **kwargs):
        """ """

        ax.plot(freqs, response, **kwargs)

        # draw transition band rectangles
        edges = self.cutoff - 0.5 * self.width
        bottom, top = ax.get_ylim()
        height = top - bottom
        rects = [Rectangle((e, bottom), self.width, height, 
                 facecolor=cutcolor, alpha=trans_alpha) for e in edges]
        [ax.add_patch(rect) for rect in rects]

        #add frequency cutoff lines
        [ax.axvline(x=c, color=cutcolor) for c in self.cutoff]

        return ax



        

    def plot(self, size=(8,6), gridalpha=0.3, worN=512):
        """ """

        fig, axarr = plt.subplots(3, 1, figsize=size)
        
        #Impulse response axis

        #

        plt.tight_layout()
        plt.show()



class IIR(abc.ABC, FilterViewer, mixins.ViewInstance):
    """Base class for infinite impulse response filters.

    This base class defines common and expected methods used by all concrete
    IIR filters in the iir module. It is not instantiable and all concrete
    IIRs must override all abstract methods.
    """

    # FIXME initialize with a pass and stop band edge frequencies not cutoff
    # and width. Note btype can now be inferred and is not needed. Also note
    # this change will allow for asymetrical transitions on bandpass and
    # bandstop filters that we could not do when specifying a single width!
    def __init__(self, fpass, fstop, fs, btype, pass_db, stop_db, fmt):
        """Initialize this IIR filter.

        Args:
            cutoff: float or 2-el sequence
                Frequency or Frequencies at which the filter gain drops to
                -6 dB or 1/2 of the signals amplitude.
            width: float
                The width of the transition between each pass and stop band
                in this filter.
            fs: int
                The sapling rate of the signal to be filtered.
            btype: str
                A filter type that must match one of 'lowpass', 'highpass',
                'bandpass' or 'bandstop'. If specifying a band filter cutoff
                must be a 2-el sequence.
            pass_db: float
                The maximum amplitude loss in the passband. Default is 1 dB
                corresponding to a max amplitude loss of 1%.
        if depahse:
            stop_db: float
                The minimum attenuation in the stop band. Default is 40 dB
                corresponding to an amplitude attenuation of 100.
            fmt: str
                A scipy filter format str. specifying the format of this
                filters coeffecients. Must be one of 'ba', 'zpk', or 'sos'.
                Default is to use second order sections 'sos'. This is
                encouraged this leads to stable filters when the order of
                the filter is high or the transition width is narrow. Scipy
                does not have filters for zpk format so these are converted
                to sos.
        """

        self.fs = fs
        self.nyq = fs/2
        self.cutoff = np.atleast_1d(cutoff)
        self.norm_cutoff = self.cutoff / self.nyq
        self.width = width
        self.btype = btype
        self.pass_db = pass_db
        self.stop_db = stop_db
        self.fmt = 'sos' if fmt == 'zpk' else fmt
        # on subclass init build the filter
        self.coeffs = self._build()
        
    @property
    def ftype(self):
        """Returns the scipy name of this IIR filter.

        Please see scipy.signal.iirfilter ftype arg for full listing of
        available scipy filters.
        """

        return type(self).__name__.lower()

    @abc.abstractproperty
    def order(self):
        """Returns filter order for a concrete IIR filter needed to reach
        this filters attenuation criteria."""

    def edges(self):
        """Returns the pass & stop band edge frequencies of this filter."""

        # FIXME Remove this as the pass and stop bands will be provided
        w = self.width
        widths = {'lowpass': w/2, 
                  'highpass': -w/2, 
                  'bandpass': np.array([-w, w])/2, 
                  'bandstop': np.array([w, -w])/2}
       
        return self.cutoff, self.cutoff + widths[self.btype] * 2 # FIXME 2

    def _build(self):
        """Designs a digital filter of a given order that meets pass and
        stop band attenuation criteria in transition bands around the cutoff
        frequencies.

        Returns: ndarray of filter coeffecients in specified scipy 'fmt'. 
        """

        """
        return sps.iirfilter(self.order, self.cutoff, rp=self.pass_db,
                btype=self.btype, ftype=self.ftype, output=self.fmt,
                fs=self.fs)
        """

        """
        return sps.butter(*self.order, output='sos', fs=self.fs)
        """

        # FIXME no longer need edges just pass in fpass and fstop here 
        print(self.edges())
        return sps.iirdesign(*self.edges(), gpass=self.pass_db, 
                             gstop=self.stop_db, ftype=self.ftype, 
                             output=self.fmt, fs=self.fs)

    def apply(self, data, chunksize, axis, dephase=True, zi=None, **kwargs):
        """Apply this filter to an ndarray or producer of ndarrays.

        Args:
            data: ndarray or producer of ndarrays
                The data to be filtered.
            chunksize: int
                The number of samples to hold in memory during filtering.
            axis: int
                The axis of data along which to apply the filter. If data is
                multidimensional, the filter will be independently applied
                along all slices along axis.
            dephase: bool
                If True the phase delay introduced by this filter will be
                removed by applying the filter in the forwards and backwards
                direction along data axis. If False, the filtered output  
                will be delayed relative to data.
            zi: ndarray
                An array of initial conditions (i.e. steady-state responses)
                whose shape depends on the filter coeffecient format. For 
                'sos' format, zi has shape nsections x (...,2,...) where 
                (...,2,...) has the same shape as data but with 2 along 
                axis. This shape is because each section of the sos has a 
                delay of 2 along axis. For more information see lfilter_zi 
                and sosfilt_zi in scipy's signal module. This argument is 
                ignored if dephase is True. 
            kwargs: dict
                Any valid keyword argument for the producer constructor.
                Please type help(openseize.producer) for more details.

        Returns:
            An ndarray of filtered data or a producer of ndarrays of
            filtered data. The output type will match the input data type.
        """

        pro = producer(x, chunksize, axis, **kwargs)

        if self.fmt == 'sos':
            if depahse:
                filtfunc = nm.sosfiltfilt
            else:
                filtfunc = nm.sosfilt

        if self.fmt == 'ba':
            if dephase:
                filtfunc = nm.filtfilt
            else:
                filtfunc = nm.lfilter
        
        result = filtfunc(pro, self.coeffs, chunksize, axis)
        
        if isinstance(x, np.ndarray):
            # if data is an ndarray return an ndarray
            result = np.concatenate([arr for arr in result], axis=axis)

        return result





                





