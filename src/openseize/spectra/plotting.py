"""Tools for plotting power spectrums and short-time Fourier transforms.

This module contains the following classes and functions:

- banded: A function for plotting bands (e.g. Confidence Intervals) onto
    a PSD estimate.

- STFTViewer: An iteractive matplotlib figure for viewing the STFT of
    multichannel EEG data.
"""

from functools import partial
from typing import Optional, Sequence, Tuple

from matplotlib import widgets
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from openseize.core.arraytools import nearest1D
from openseize.core.arraytools import slice_along_axis


def banded(x: npt.NDArray[np.float64],
           upper: npt.NDArray[np.float64],
           lower: npt.NDArray[np.float64],
           ax: plt.Axes,
           **kwargs,
) -> plt.Axes:
    """Plots upper & lower error bands on an existing axis.

    Args:
        x:
            1-D array of abscissa (x-axis) values.
        upper:
            1-D array of upper error bounds to plot (e.g. STD, SEM, or CI)
        lower:
            1-D array of lower error bounds to plot (e.g. STD, SEM, or CI)
        ax:
            A Matplotlib axis instance to plot x and y onto.
        **kwargs:
            Any valid kwarg for matplotlib's plot and fill_between funcs.

    Returns:
        A matplotlib axis instance.
    """

    x = np.arange(len(upper)) if x is None else x

    color = kwargs.pop('color', 'k')
    facecolor = kwargs.pop('facecolor', 'tab:gray')
    alpha = kwargs.pop('alpha', 0.4)

    ax.fill_between(x, lower, upper, color=color, facecolor=facecolor,
                    alpha=alpha, **kwargs)
    return ax


# This viewer needs to track may plot attributes
# pylint: disable-next=too-many-instance-attributes
class STFTViewer:
    """An interactive matplotlib figure for plotting the magnitude of
    a Short-Time Fourier transform of multichannel eeg data."""

    def __init__(self,
                 freqs: npt.NDArray[np.float64],
                 time: npt.NDArray[np.float64],
                 data: npt.NDArray[np.float64],
                 scale: Optional[str] = 'dB',
                 chs: Optional[Sequence[int]] = None,
                 names: Optional[Sequence[str]] = None,
                 stride: int = 120,
                 figsize: Tuple[int, int] = (8,6)
    ) -> None:
        """Initialize this Viewer by creating the matploltib figure.

        Args:
            data:
                The squared norm of the STFT array. Must have shape
                (channels, frequencies, time).
            freqs:
                1-D array of STFT frequencies in Hz.
            time:
                1-D array of STFT times in secs.
            scale:
                String specifying a scaling function to apply to data prior
                to display. Default is the dB power scale.
            chs:
                A sequence of channel indices to display. None plots all
                channels.
            names:
                A sequence of channel names to adorn subplots with.
            stride:
                The amount of data in secs to display centered on the
                current time. The default of 120 secs displays +/- 60s
                around current time.
            figsize:
                A shape tuple for the displayed matplotlib figure.

        Examples:
            >>> # Compute the STFT of the demo data
            >>> # import demo data and make a producer
            >>> from openseize.demos import paths
            >>> from openseize.file_io.edf import Reader
            >>> from openseize import producer
            >>> from openseize.spectra.estimators import stft
            >>> from openseize.spectra.plotting import STFTViewer
            >>> fp = paths.locate('recording_001.edf')
            >>> reader = Reader(fp)
            >>> pro = producer(reader, chunksize=10e4, axis=-1)
            >>> # Compute the STFT of the demo data
            >>> freqs, time, estimate = stft(pro, fs=5000, axis=-1)
            >>> STFTViewer(freqs, time, estimate, chs=[0,1,2])
        """

        self.freqs = freqs
        self.time = time
        self.data = self.rescale(data, scale)
        self.scale = scale
        self.chs = self.init_channels(chs)
        self.names = self.init_names(names)
        self.stride = stride
        self.current = stride / 2

        # min and max values for each channel
        self.vmins = np.amin(self.data, axis=(1,2))
        self.vmaxes = np.amax(self.data, axis=(1,2))

        # initialize viewer to display all frequencies
        self.limits = (freqs[0], freqs[-1])

        # initialize a configured figure to contain this viewer
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
        self.add_low_limit(self.axarr[0])
        self.add_high_limit(self.axarr[0])

        plt.ion()
        plt.show()

    def rescale(self, data, scale):
        """Rescales the data for easier visualization."""

        if scale is None:
            return data

        if scale == 'dB':
            return 10 * np.log10(data + 1)

        raise ValueError('Unknown scaling')

    def init_channels(self, chs):
        """Initialize the channels to display in this viewer."""

        if not chs:
            chs = np.arange(self.data.shape[0])

        return chs

    def init_names(self, names):
        """Initialize the channel names to display in this viewer."""

        if not names:
            names = [f'Ch {x}' for x in self.chs]

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
        vmax = self.time[-1] - self.stride // 2
        self.slider = widgets.Slider(self.slider_ax, 'Time', vmin,
                                     vmax, valinit=self.current,
                                     valstep=self.stride)

        # define the callback of this slider
        self.slider.on_changed(self.slide)

    # pylint: disable-next=unused-argument
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

    # pylint: disable-next=unused-argument
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

    # pylint: disable-next=unused-argument
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
        self.stride_entry = widgets.TextBox(self.restride_ax, 'Stride ',
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

    def add_low_limit(self, ax):
        """Add a fully configured low freq. limit entry to this viewer.

        Args:
            ax (mpl axis): A matplotlib axis instance where the low limit
                           will be added.
        """

        position = ax.get_position()
        # build entry axis container relative to first plotting axis
        left = position.x0 - .06
        bottom = position.y0 - 0.015
        self.low_limit_ax = plt.axes([left, bottom, 0.05, 0.03])

        # add textbox setting init value to init limit of plot axis
        low, _ = ax.get_ylim()
        low = str(int(low))
        self.low_limit = widgets.TextBox(self.low_limit_ax, '', low, '1',
                                         '1', textalignment='right')
        # define the callback of this limit entry
        self.low_limit.on_submit(self.limit_submit)

    def add_high_limit(self, ax):
        """Add a fully configured high freq. limit entry to this viewer.

        Args:
            ax (mpl axis): A matplotlib axis instance where the high limit
                           will be added.
        """

        position = ax.get_position()
        # build entry axis container relative to first plotting axis
        left = position.x0 - .06
        top = position.y1 - 0.015
        self.high_limit_ax = plt.axes([left, top, 0.05, 0.03])

        # add textbox setting init value to init limit of plot axis
        _, high = ax.get_ylim()
        high = str(int(high))
        self.high_limit = widgets.TextBox(self.high_limit_ax, '', high, '1',
                                          '1', textalignment='right')
        # define the callback of this limit entry
        self.high_limit.on_submit(self.limit_submit)

    # pylint: disable-next=unused-argument
    def limit_submit(self, value):
        """On freq. limit change update the stored limits & update plot."""

        low, high = int(self.low_limit.text), int(self.high_limit.text)
        self.limits = (low, high)
        self.update()

    def fmt_coord(self, channel, x, y):
        """When hovering over an axis display the time, freq & stft
        magnitude for each subplot."""

        # get nearest freq and time indices & stft scaled magnitude
        f_idx = nearest1D(self.freqs, y)
        t_idx = nearest1D(self.time, x)
        z = self.data[channel, f_idx, t_idx]
        scale = self.scale if self.scale else ''

        msg = '\ntime = {:.2f}, freq = {:.2f}, [{:.1f}] {scale}\n {blank:>100}'
        return msg.format(x, y, z, scale=scale, blank='_')

    def update(self):
        """Updates the data displayed to this viewer plotting axes."""

        # pylint: disable-next=expression-not-assigned
        [ax.clear() for ax in self.axarr]

        # get data for channels to display
        x = self.data[self.chs]

        # slice the frequency vector and data along 2nd (freq) axis
        low_f = nearest1D(self.freqs, self.limits[0])
        high_f = nearest1D(self.freqs, self.limits[1])
        f = slice_along_axis(self.freqs, low_f, high_f + 1)
        x = slice_along_axis(x, low_f, high_f + 1, axis=-2)

        # slice the time vector & data along last (time) axis
        time_a = nearest1D(self.time, self.current - self.stride / 2)
        time_b = nearest1D(self.time, self.current + self.stride / 2)
        x = slice_along_axis(x, time_a, time_b, axis=-1)
        t = slice_along_axis(self.time, time_a, time_b)

        for idx, ch in enumerate(self.chs):

            # fetch subplot axis and display sliced data
            ax = self.axarr[idx]
            vmin, vmax = self.vmins[idx], self.vmaxes[idx]
            ax.pcolormesh(t, f, x[idx], shading='nearest', vmin=vmin,
                          vmax=vmax, rasterized=True)

            # configure ticks
            ax.xaxis.set_visible(False)

            # configure the string fmt for this axis disp at top right
            ax.format_coord = partial(self.fmt_coord, ch)

        # add labels to last axis
        self.axarr[-1].set_ylabel('Frequency (Hz)', fontsize=12)
        self.axarr[-1].set_xlabel('Time (s)', fontsize=12)
        self.axarr[-1].xaxis.set_visible(True)

        # add channel names
        for ax, name in zip(self.axarr, self.names):
            ax.annotate(name, (0.95, .85), xycoords='axes fraction',
                        color='white', fontsize=12)

        # update drawn data
        plt.draw()
