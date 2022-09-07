import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets

from openseize.core.arraytools import nearest1D, slice_along_axis

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
        self.start = 0
        self.stride = stride
        self.ylim = (freqs[0], freqs[-1])
        if not names:
            self.names = ['Ch {}'.format(x) for x in self.chs]
        else:
            self.names = names

        #create figure, axes
        self.fig, self.axarr = plt.subplots(len(self.chs), 1,
                                            figsize=figsize, sharex=True,
                                            sharey=True)
        self.fig.subplots_adjust(left=0.08, bottom=0.2, right=.98, top=0.98)

        #add a slider widget
        self.slider_ax = plt.axes([0.15, 0.08, 0.73, 0.03])
        self.slider = widgets.Slider(self.slider_ax, label='Time',
                                      valmin=0, valmax=time[-1],
                                      valinit=self.start, valstep=stride)
        self.slider.on_changed(self.slide)

        #add time text widget
        self.time_txt_ax = plt.axes([.89, 0.08, 0.03, 0.03])
        self.time_txt = widgets.TextBox(self.time_txt_ax, '', initial='0',
                                      textalignment='center',
                                      color='white')

        # make initial draw to axes
        self.update()

        #add a low freq text widget
        self.flow_ax = plt.axes([0.02, .74, 0.05, 0.03])
        low, high = self.axarr[0].get_ylim()
        self.flow_text = widgets.TextBox(self.flow_ax, '',
                                         initial=str(int(low)),
                                         textalignment='right', color='1',
                                         hovercolor='0.90')
        self.flow_text.on_submit(self.ylim_submit)
        
        #add a high freq text widget
        self.fhigh_ax = plt.axes([0.02, .96, 0.05, 0.03])
        self.fhigh_text = widgets.TextBox(self.fhigh_ax, '',
                                          initial=str(int(high)), 
                                          textalignment='right', color='1',
                                          hovercolor='0.90')
        self.fhigh_text.on_submit(self.ylim_submit)

        plt.ion()
        plt.show()

    
    def slide(self, value):
        """On slider movement update the start time & update plot."""

        self.start = int(self.slider.val)
        self.time_txt.set_val(self.start)
        self.update()


    def ylim_submit(self, value):
        """ """

        low, high = int(self.flow_text.text), int(self.fhigh_text.text)
        self.ylim = (low, high)
        self.update()


    def update(self):
        """ """

        [ax.clear() for ax in self.axarr]

        a = nearest1D(self.time, self.start - self.stride / 2) 
        b = nearest1D(self.time, self.start + self.stride / 2)
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
    from openseize.spectra.estimators import stft

    fp = demos.paths.locate('recording_001.edf')
    reader = Reader(fp)
    pro = producer(reader, chunksize=10e6, axis=-1)
    dpro = downsample(pro, M=25, fs=5000, chunksize=10e6, axis=-1)
    freqs, time, Z = stft(dpro, fs=200, axis=-1, asarray=True)

    viewer = StftViewer(freqs, time, Z, chs=[0, 1, 2], 
                        names=['LFC', 'RVC', 'LSC'])

