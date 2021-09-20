from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import freqz
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# as you add filters go ahead and make a superclass with common meths

class Butterworth:
    """Creates an IIR Butterworth filter.

    Attrs:
        order (int),              the max delay in samples used to create each 
                                  output -- default is 8
        filter_type (str),        one of {'lowpass', 'highpass', 'bandpass',
                                  'bandstop'} -- default is bandpass
        critical_freqs (int/seq), the cut-off frequencies of the filter
                                  --default is [2,10]
        sampling_freq (int),      the sampling frequency of data acquistion
                                  --default is 20 Hz
    """
    def __init__(self, order=8,  f_type='bandpass', critical_freqs=[2,10],
                 sampling_freq=20):
        self.order = order
        self.f_type = f_type
        self.nyquist = 0.5 * sampling_freq
        self.critical_freqs = [freq/self.nyquist for freq in critical_freqs]
        self.samping_freq = sampling_freq
        
        # numerator/denominator of the z-transform polys
        self._b, self._a = butter(self.order, self.critical_freqs,
                               btype=f_type, analog=False, output='ba')

    def plot(self, **kwargs):
        """Plot the gain of the filter as a function of the frequencies.

        The frequency response is given by the H(z) transform --see
        https://en.wikipedia.org/wiki/Digital_filter.

        kwargs passed to freqz --see scipy's freqz documentation
        """
        plt.figure()
        # w is radians/sample and h is complex frequency response
        w, h = freqz(self._b, self._a, **kwargs)
        # convert w to Hz and plot
        plt.plot((self.nyquist/np.pi) * w, abs(h), 
                  label='order = %d' % self.order)
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.legend(loc='best')
        plt.show()
        
    def filter(self, data):
        """Apply this filter to data.
        
        Args:
            data, n-el sequence of values to filter
        """
        return filtfilt(self._b, self._a, data)


if __name__ == '__main__':
    """Self test code..."""
    # create a filter
    butter = Butterworth(order=8, f_type='bandpass', critical_freqs=[0.5,20], sampling_freq=1000)
    # show the filters gain
    butter.plot()

    # generate sample data
    time = 1
    sample_rate = butter.samping_freq
    num_samples = int(time * sample_rate)
    t = np.linspace(0, time, num_samples)
    
    # make a small 10 Hz riding on top of a larger 100 Hz sinusoidal signal
    x = 0.25 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t)
    plt.figure(2)
    plt.plot(t, x, label='Before Filtering')
    plt.legend(loc='upper right')

    #filter data
    y = butter.filter(x)
    plt.figure(3)
    plt.plot(t, y, label='After Filtering')
    plt.legend(loc='upper right')
    plt.show()

