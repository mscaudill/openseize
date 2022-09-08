import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets

from openseize.core.arraytools import nearest1D, slice_along_axis
from openseize.spectra.metrics import power_norm

def banded(x, upper, lower, ax, **kwargs):
    """Plots upper & lower error bands on an existing axis.

    Args:
        x: 1D array
            Abscissa values for each each value in y ordinates.
        upper: 1D array
            Upper error bounds to plot (e.g. STD, SEM, or CI bound)
        lower: 1D array
            Lower error bounds to plot (e.g. STD, SEM, or CI bound)
        ax: axis instance
            A Matplotlib figure axis instance to plot x and y to. If None
            (Default) creates a new axis instance.
        **kwargs: dict
            Any valid kwarg for matplotlib's plot and fill_between funcs.

    Returns: A matplotlib axis instance.
    """

    x = np.arange(len(arr)) if x is None else x

    color = kwargs.pop('color', 'k')
    facecolor = kwargs.pop('facecolor', 'tab:gray')
    alpha = kwargs.pop('alpha', 0.4)

    ax.fill_between(x, lower, upper,color='k', facecolor=facecolor,
            alpha=alpha, **kwargs)
    return ax   


class StftViewer:
    """An interactive matplotlib figure for plotting the magnitude of
    a Short-Time Fourier transform.

    Attrs:
        data:
        freqs:
        time:
        chs:
        kwargs:
    """

    def __init__(self, freqs, time, data, chs=None, stride=30,
                 figsize=(8,6), names=None, **kwargs):
        """Initialize this Viewer by creating the matploltib figure."""

        self.freqs = freqs
        self.time = time
        self.data = np.real(data)**2 + np.imag(data)**2
        self.chs = np.arange(self.data.shape[0]) if not chs else chs
        self.stride = stride
        self.current = stride / 2
        self.ylim = (freqs[0], freqs[-1])
        if not names:
            self.names = ['Ch {}'.format(x) for x in self.chs]
        else:
            self.names = names

        #create figure, axes
        self.fig, self.axarr = plt.subplots(len(self.chs), 1,
                                            figsize=figsize, sharex=True,
                                            sharey=True)
        self.axarr = np.atleast_1d(self.axarr)
        self.fig.subplots_adjust(left=0.08, bottom=0.2, right=.98, top=0.98)

        #add a slider widget
        self.slider_ax = plt.axes([0.15, 0.08, 0.73, 0.03])
        self.slider = widgets.Slider(self.slider_ax, label='Time',
                                      valmin=self.stride//2, 
                                      valmax=time[-1]-self.stride//2,
                                      valinit=self.current, valstep=stride)
        self.slider.on_changed(self.slide)

        #add time text widget
        self.time_txt_ax = plt.axes([.89, 0.08, 0.1, 0.03])
        initval = str(stride//2)
        self.time_txt = widgets.TextBox(self.time_txt_ax, '',
                                        initial=initval,
                                        textalignment='left', 
                                        color='white')
        self.time_txt.on_submit(self.time_submit)
        
        # make initial draw to axes
        self.update()

        #add a low freq text widget
        pos = self.axarr[0].get_position()
        left = pos.x0 - .06
        bottom = pos.y0 - 0.015
        top = pos.y1 - 0.015
        self.flow_ax = plt.axes([left, bottom, 0.05, 0.03])
        low, high = self.axarr[0].get_ylim()
        self.flow_text = widgets.TextBox(self.flow_ax, '',
                                         initial=str(int(low)),
                                         textalignment='right', color='1',
                                         hovercolor='0.90')
        self.flow_text.on_submit(self.ylim_submit)
        
        #add a high freq text widget
        self.fhigh_ax = plt.axes([left, top, 0.05, 0.03])
        self.fhigh_text = widgets.TextBox(self.fhigh_ax, '',
                                          initial=str(int(high)), 
                                          textalignment='right', color='1',
                                          hovercolor='0.90')
        self.fhigh_text.on_submit(self.ylim_submit)

        plt.ion()
        plt.show()

    
    def slide(self, value):
        """On slider movement update the current time & update plot."""

        self.current = int(self.slider.val)
        self.time_txt.set_val(self.current)
        self.update()

    def time_submit(self, value):
        """On time submission jump to that time in seconds."""

        value = int(value)
        if value < self.stride // 2:
            value = self.stride // 2
            self.time_txt.set_val(value)

        elif value > self.data.shape[-1] - self.stride // 2:
            value = self.data.shape[-1] - self.stride // 2
            self.time_txt.set_val(value)

        self.slider.set_val(value)


    def ylim_submit(self, value):
        """ """

        low, high = int(self.flow_text.text), int(self.fhigh_text.text)
        self.ylim = (low, high)
        self.update()


    def update(self):
        """ """

        [ax.clear() for ax in self.axarr]

        a = nearest1D(self.time, self.current - self.stride / 2) 
        b = nearest1D(self.time, self.current + self.stride / 2)
        x = self.data[self.chs]
        x = slice_along_axis(x, a, b, axis=-1)
        t = slice_along_axis(time, a, b)

        vmin = np.amin(x)
        vmax = np.amax(x)

        for idx, ch in enumerate(self.chs):

            self.axarr[idx].pcolormesh(t, self.freqs, x[idx],
            shading='nearest', vmin=vmin, vmax=vmax, rasterized=True)
            self.axarr[idx].xaxis.set_visible(False)
        
        [ax.set_ylim(*self.ylim) for ax in self.axarr]
        self.axarr[-1].set_ylabel('Frequency (Hz)', fontsize=12)
        for ax, name in zip(self.axarr, self.names):
            ax.annotate(name, (0.95, .85), xycoords='axes fraction',
                        color='white', fontsize=12)
        self.axarr[-1].set_xlabel('Time (s)', fontsize=12)
        self.axarr[-1].xaxis.set_visible(True)
        plt.draw()



if __name__ == '__main__':

    from openseize import demos
    from openseize import producer
    from openseize.io.edf import Reader
    from openseize.resampling.resampling import downsample
    from openseize.filtering.iir import Notch
    from openseize.spectra.estimators import stft
    from openseize.spectra.metrics import power_norm, power

    #fp = demos.paths.locate('recording_001.edf')
    fp ='/media/matt/Magnus/data/eigensort/'+\
    'DL00A1_P043_nUbe3a_15_53_3dayEEG_2019-04-03_13_41_20.edf'
    reader = Reader(fp)
    pro = producer(reader, chunksize=10e6, axis=-1)
    
    #downsample data
    dpro = downsample(pro, M=25, fs=5000, chunksize=10e6, axis=-1)
    
    freqs, time, Z = stft(dpro, fs=200, axis=-1, asarray=True)
    data = np.real(Z)**2 + np.imag(Z)**2
    
    # easy to notch
    line_freq = nearest1D(freqs, 60)
    data[:, line_freq-4:line_freq+4, :] = 0

    # get average power in each time bin
    #fnorm = power(data, freqs, None, None, axis=-1)
    #fnorm = np.expand_dims(fnorm, axis=-1)
    #norm = np.mean(fnorm, axis=-1, keepdims=True)
    #normed = data / fnorm

    normed = power_norm(data, freqs, axis=1)
    viewer = StftViewer(freqs, time, normed, chs=[0, 1, 2], 
                        names=['LFC', 'RVC', 'LSC'])

