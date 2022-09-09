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
        self.data = data
        self.chs = self.init_channels(chs)
        self.names = self.init_names(names)
        self.stride = stride
        self.current = stride / 2

        # initialize viewer to display all frequencies
        self.limits = (freqs[0], freqs[-1])

        # intialize a configured figure to contain this viewer
        self.init_figure(figsize)

        # add a configured widgets to this viewers figure
        self.add_slider()
        self.add_time()
        self.add_forward()
        self.add_reverse()
        self.add_stride()

        # make initial draw to axes
        self.update()

        # add low & high frequency limits
        ax0_pos = self.axarr[0].get_position()
        self.add_low_limit(ax0_pos)
        self.add_high_limit(ax0_pos)
        
        plt.ion()


    def init_channels(self, chs):
        """Initialize the channels to display in this viewer."""

        if not chs:
            chs = np.arange(self.data.shape[0])
        
        return chs


    def init_names(self, names):
        """Initialize the channel names to display in this viewer."""

        if not names:
            names = ['Ch {}'.format(x) for x in self.chs]
        
        return names


    def init_figure(self, figsize):
        """Initialize the figure containing this viewer."""

        # create figure, mpl axes array, & config. margins
        nrows = len(self.chs)
        self.fig, self.axarr = plt.subplots(nrows, 1, figsize=figsize,
                                            sharex=True, sharey=True)
        self.axarr = np.atleast_1d(self.axarr)
        self.fig.subplots_adjust(left=0.08, bottom=0.2, right=.98, top=0.98)


    def add_slider(self):
        """Adds a fully configured slider widget to this viewer's figure."""

        # build slider container axis
        self.slider_ax = plt.axes([0.15, 0.08, 0.73, 0.03])
        
        # add slider widget setting its min, max & step
        vmin = self.stride // 2
        vmax = time[-1] - self.stride // 2
        step = self.stride
        self.slider = widgets.Slider(self.slider_ax, 'Time', vmin, 
                                     vmax, valinit=self.current, 
                                     valstep=self.stride)
        
        # define the callback of this slider
        self.slider.on_changed(self.slide)


    def slide(self, value):
        """On slider movement update current & update plot."""

        self.current = int(self.slider.val)
        self.time_entry.set_val(self.current)
        self.update()


    def add_time(self):
        """Adds a fully configured time entry box to this viewer."""

        # build container axis
        self.time_ax = plt.axes([.89, 0.08, 0.1, 0.03])

        # add textbox setting its initial value
        initval = str(self.stride//2)
        self.time_entry = widgets.TextBox(self.time_ax, '', initval, '1',
                                        textalignment='left')

        # define the callback of this time entry
        self.time_entry.on_submit(self.time_submission)


    def time_submission(self, value):
        """On time submission update current & update plot."""

        value = int(value)

        #value must be in range [stride //2 , total time - stride // 2] 
        if value < self.stride // 2:
            value = self.stride // 2

        elif value > self.data.shape[-1] - self.stride // 2:
            value = self.data.shape[-1] - self.stride // 2

        self.time_entry.set_val(value)
        self.slider.set_val(value)


    def add_forward(self):
        """Add a fully configured time advance button to this viewer."""

        # build container axis, add widget and set callback
        self.forward_ax = plt.axes([0.84, .03, .04, 0.04])
        self.forward_button = widgets.Button(self.forward_ax, '>')
        self.forward_button.label.set_fontsize(15)
        self.forward_button.on_clicked(self.forward)
    

    def forward(self, event):
        """On forward button press advance current by 1 stride & update."""

        # maximum is total time - 1/2 a stride
        if self.current > self.data.shape[-1] - self.stride // 2:
            self.current = self.data.shape[-1] - self.stride // 2
        
        else:
            self.current += self.stride
        
        self.slider.set_val(self.current)
   

    def add_reverse(self):
        """Add a fully configured time reverse button to this viewer."""

        # build container axis, add widget and set callback
        self.reverse_ax = plt.axes([0.15, .03, .04, 0.04])
        self.reverse_button = widgets.Button(self.reverse_ax, '<')
        self.reverse_button.label.set_fontsize(15)
        self.reverse_button.on_clicked(self.reverse)
    

    def reverse(self, event):
        """On reverse button press regress current by 1 stride & update."""
        
        # minimum is 1/2 a stride
        if self.current < self.stride // 2:
            self.current = self.stride // 2
        
        else:
            self.current -= self.stride
        
        self.slider.set_val(self.current)


    def add_stride(self):
        """Add a fully configured entry to change the stride amount
        displayed to this viewer."""

        # build container axis
        self.restride_ax = plt.axes([.45, 0.03, 0.04, 0.03])

        # add textbox setting its initial value
        self.stride_entry = widgets.TextBox(self.restride_ax, 'Stride',
                                            self.stride, '1', 
                                            textalignment='center')

        # define the callback of this time entry
        self.stride_entry.on_submit(self.stride_submission)


    def stride_submission(self, value):
        """On stride submission update stride and update plot."""

        value = int(value)
        
        self.stride = value if value > 0 else self.stride
        self.current = self.stride // 2

        # on stride change the sliders min, max must change
        self.slider.valmin = self.stride // 2
        self.slider.valmax = self.data.shape[-1] - self.stride // 2

        self.slider.set_val(self.current)
        self.update()

    
    def add_low_limit(self, position):
        """Add a fully configured low freq. limit entry to this viewer.
        
        Args:
            position (mpl bbox): A bounding box for the first axis displayed
                                 in this viewer.
        """

        # build entry axis container relative to first plotting axis 
        left = position.x0 - .06
        bottom = position.y0 - 0.015
        self.low_limit_ax = plt.axes([left, bottom, 0.05, 0.03])
        
        # add textbox setting init value to init limit of plot axis
        low, _ = self.axarr[0].get_ylim()
        low = str(int(low))
        self.low_limit = widgets.TextBox(self.low_limit_ax, '', low, '1',
                                         '1', textalignment='right')
        # define the callback of this limit entry
        self.low_limit.on_submit(self.limit_submit)

    
    def add_high_limit(self, position):
        """Add a fully configured high freq. limit entry to this viewer.
        
        Args:
            position (mpl bbox): A bounding box for the first axis displayed
                                 in this viewer.
        """

        # build entry axis container relative to first plotting axis 
        left = position.x0 - .06
        top = position.y1 - 0.015
        self.high_limit_ax = plt.axes([left, top, 0.05, 0.03])

        # add textbox setting init value to init limit of plot axis
        _, high = self.axarr[0].get_ylim()
        high = str(int(high))
        self.high_limit = widgets.TextBox(self.high_limit_ax, '', high, '1',
                                          '1', textalignment='right')
        # define the callback of this limit entry
        self.high_limit.on_submit(self.limit_submit)


    def limit_submit(self, value):
        """On freq. limit change update the stored limits & update plot."""

        low, high = int(self.low_limit.text), int(self.high_limit.text)
        self.limits = (low, high)
        self.update()


    def update(self):
        """Updates the data displayed to this viewer plotting axes."""

        [ax.clear() for ax in self.axarr]

        # slice the freqs, time vector & data around current time
        a = nearest1D(self.time, self.current - self.stride / 2) 
        b = nearest1D(self.time, self.current + self.stride / 2)
        x = self.data[self.chs]
        x = slice_along_axis(x, a, b, axis=-1)
        t = slice_along_axis(self.time, a, b)

        low = nearest1D(self.freqs, self.limits[0])
        high = nearest1D(self.freqs, self.limits[1])
        f = slice_along_axis(self.freqs, low, high+1)
        x = slice_along_axis(x, low, high+1, axis=-2)

        for idx, ch in enumerate(self.chs):

            self.axarr[idx].pcolormesh(t, f, x[idx], shading='nearest', 
                                       rasterized=True)
            self.axarr[idx].xaxis.set_visible(False)
        
        #[ax.set_ylim(*self.limits) for ax in self.axarr]
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

    fp = demos.paths.locate('recording_001.edf')
    #fp ='/media/matt/Magnus/data/eigensort/'+\
    #'DL00A1_P043_nUbe3a_15_53_3dayEEG_2019-04-03_13_41_20.edf'
    reader = Reader(fp)
    pro = producer(reader, chunksize=10e6, axis=-1)
    
    #downsample data
    dpro = downsample(pro, M=25, fs=5000, chunksize=10e6, axis=-1)
    
    freqs, time, Z = stft(dpro, fs=200, axis=-1, asarray=True)
    data = np.real(Z)**2 + np.imag(Z)**2
    
    # easy to notch
    line_freq = nearest1D(freqs, 60)
    data[:, line_freq-5:, :] = 0

    # get average power in each time bin
    #fnorm = power(data, freqs, None, None, axis=-1)
    #fnorm = np.expand_dims(fnorm, axis=-1)
    #norm = np.mean(fnorm, axis=-1, keepdims=True)
    #normed = data / fnorm

    #normed = power_norm(data, freqs, axis=1)
    viewer = StftViewer(freqs, time, data, chs=[0, 1, 2], 
                        names=['LFC', 'RVC', 'LSC'])

