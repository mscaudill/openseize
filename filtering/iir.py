"""A collection of callable Infinite Impulse Response filters.

The collection includes Butterworth, Chebyshev Type I, Chebyshev Type II and 
Elliptical filters. Each filter inherits the ability to plot itself and be
applied to ndarrays or producers of ndarrays from the IIR Base class 
(see openseize.filtering.bases).

Typical usage example:

    # build a Chebyshev Type I filter that bandpass filters 500 to 500 Hz
    # with transition bands of 100 Hz on each side of the passband
    cheby1 = Cheby1(fpass=[500, 700], fstop=[400, 800], fs=fs)

    #plot the filters impulse and frequency responses
    cheby1.plot()

    # finally apply the filter to a producer of ndarrays called 'data'
    filtered = cheby1(data, chunksize=10e6, axis=-1, dephase=True)

    # process each filtered data chunk
    for arr in filtered:
        # do some processing

"""

import numpy as np
import scipy.signal as sps

from openseize.filtering.bases import IIR

class Butter(IIR):
    """A callable digital Butterworth IIR filter.

    Designs a minimum order Butterworth filter that meets both pass and
    stop band attenuation criteria.

    Attrs:
        fpass, fstop: float or 1-D array
            Pass and stop band edge frequencies (Hz).
            For example:
                - Lowpass: fpass = 1000, fstop = 1100
                - Highpass: fpass = 2000, fstop = 1800 
                - Bandpass: fpass = [400, 800], fstop = [300, 900]
                - Bandstop: fpass = [100, 200], fstop = [120, 180]
        fs: int
            The sampling rate of the data to be filtered.
        gpass: float
            The maximum loss in the pass band (dB). Default is 1.0 dB which
            is an amplitude loss of ~ 11%.
        gstop: float
            The minimum attenuation in the stop band (dB). The default is 40
            dB which is an amplitude loss of 99%
        fmt: str
            A scipy filter format str. specifying the format of this
            filter's coeffecients. Must be one of 'ba', 'zpk', or 'sos'.
            Default is to use second order sections 'sos'. SOS format is
            encouraged as it is more stable than the transfer function fmt
            ('ba') when the filter order is high or the transition widths
            are narrow.

    This filter is callable meaning after initialization it can be called
    like a function to apply the filter to data. For additional details 
    about call arguments type help(Butter.call)
    """

    def __init__(self, fpass, fstop, fs, gpass=1.0, gstop=40, fmt='sos'):
        """Initialize this Butterworth IIR filter."""

        super().__init__(fpass, fstop, gpass, gstop, fs, fmt)

    @property
    def order(self):
        """Returns the minimum order and 3 dB point that meets the pass and
        stop band design specifications."""

        return sps.buttord(self.fpass, self.fstop, self.gpass, self.gstop, 
                             fs=self.fs)


class Cheby1(IIR):
    """A callable digital Chebyshev type I IIR filter.

    Designs a minimum order Chebyshev Type I filter that meets both pass
    and stop band attenuation criteria. Chebyshev I filters have faster
    roll-offs in the transition band at the expense of ripple in the pass
    band.

    Attrs:
        fpass, fstop: float or 1-D array
            Pass and stop band edge frequencies (Hz).
            For example:
                - Lowpass: fpass = 1000, fstop = 1100
                - Highpass: fpass = 2000, fstop = 1800 
                - Bandpass: fpass = [400, 800], fstop = [300, 900]
                - Bandstop: fpass = [100, 200], fstop = [120, 180]
        fs: int
            The sampling rate of the data to be filtered.
        gpass: float
            The maximum loss in the pass band (dB). Default is 1.0 dB which
            is an amplitude loss of ~ 11%.
        gstop: float
            The minimum attenuation in the stop band (dB). The default is 40
            dB which is an amplitude loss of 99%
        fmt: str
            A scipy filter format str. specifying the format of this
            filter's coeffecients. Must be one of 'ba', 'zpk', or 'sos'.
            Default is to use second order sections 'sos'. SOS format is
            encouraged as it is more stable than the transfer function fmt
            ('ba') when the filter order is high or the transition widths
            are narrow.

    This filter is callable meaning after initialization it can be called
    like a function to apply the filter to data. For additional details 
    about call arguments type help(Cheby1.call)
    """

    def __init__(self, fpass, fstop, fs, gpass=1.0, gstop=40, fmt='sos'):
        """Initialize this Chebyshev Type I IIR filter."""

        super().__init__(fpass, fstop, gpass, gstop, fs, fmt)

    @property
    def order(self):
        """Returns the minimum order and 3 dB point that meets the pass and
        stop band design specifications."""

        return sps.cheb1ord(self.fpass, self.fstop, self.gpass, self.gstop, 
                             fs=self.fs)


class Cheby2(IIR):
    """A callable digital Chebyshev type II IIR filter.

    Designs a minimum order Chebyshev Type II filter that meets both pass
    and stop band attenuation criteria. Chebyshev II filters have faster
    roll-offs in the transition band at the expense of ripple in the stop
    band.

    Attrs:
        fpass, fstop: float or 1-D array
            Pass and stop band edge frequencies (Hz).
            For example:
                - Lowpass: fpass = 1000, fstop = 1100
                - Highpass: fpass = 2000, fstop = 1800 
                - Bandpass: fpass = [400, 800], fstop = [300, 900]
                - Bandstop: fpass = [100, 200], fstop = [120, 180]
        fs: int
            The sampling rate of the data to be filtered.
        gpass: float
            The maximum loss in the pass band (dB). Default is 1.0 dB which
            is an amplitude loss of ~ 11%.
        gstop: float
            The minimum attenuation in the stop band (dB). The default is 40
            dB which is an amplitude loss of 99%
        fmt: str
            A scipy filter format str. specifying the format of this
            filter's coeffecients. Must be one of 'ba', 'zpk', or 'sos'.
            Default is to use second order sections 'sos'. SOS format is
            encouraged as it is more stable than the transfer function fmt
            ('ba') when the filter order is high or the transition widths
            are narrow.

    This filter is callable meaning after initialization it can be called
    like a function to apply the filter to data. For additional details 
    about call arguments type help(Cheby2.call)
    """

    def __init__(self, fpass, fstop, fs, gpass=1.0, gstop=40, fmt='sos'):
        """Initialize this Chebyshev Type II IIR filter."""

        super().__init__(fpass, fstop, gpass, gstop, fs, fmt)

    @property
    def order(self):
        """Returns the minimum order and 3 dB point that meets the pass and
        stop band design specifications."""

        return sps.cheb2ord(self.fpass, self.fstop, self.gpass, self.gstop, 
                             fs=self.fs)


class Ellip(IIR):
    """A callable digital Elliptical IIR filter.

    Designs a minimum order Elliptical IIR filter that meets both pass
    and stop band attenuation criteria. Elliptical filters have faster
    roll-offs in the transition band at the expense of ripple in both the
    pass and stop bands.

    Attrs:
        fpass, fstop: float or 1-D array
            Pass and stop band edge frequencies (Hz).
            For example:
                - Lowpass: fpass = 1000, fstop = 1100
                - Highpass: fpass = 2000, fstop = 1800 
                - Bandpass: fpass = [400, 800], fstop = [300, 900]
                - Bandstop: fpass = [100, 200], fstop = [120, 180]
        fs: int
            The sampling rate of the data to be filtered.
        gpass: float
            The maximum loss in the pass band (dB). Default is 1.0 dB which
            is an amplitude loss of ~ 11%.
        gstop: float
            The minimum attenuation in the stop band (dB). The default is 40
            dB which is an amplitude loss of 99%
        fmt: str
            A scipy filter format str. specifying the format of this
            filter's coeffecients. Must be one of 'ba', 'zpk', or 'sos'.
            Default is to use second order sections 'sos'. SOS format is
            encouraged as it is more stable than the transfer function fmt
            ('ba') when the filter order is high or the transition widths
            are narrow.

    This filter is callable meaning after initialization it can be called
    like a function to apply the filter to data. For additional details 
    about call arguments type help(Ellip.call)
    """

    def __init__(self, fpass, fstop, fs, gpass=1.0, gstop=40, fmt='sos'):
        """Initialize this Elliptical IIR filter."""

        super().__init__(fpass, fstop, gpass, gstop, fs, fmt)

    @property
    def order(self):
        """Returns the minimum order and 3 dB point that meets the pass and
        stop band design specifications."""

        return sps.ellipord(self.fpass, self.fstop, self.gpass, self.gstop, 
                             fs=self.fs)



if __name__ == '__main__':

    fs = 5000

    #butter = Butter(fpass=[400], fstop=[500], fs=fs) #lowpass
    #butter = Butter(fpass=[500], fstop=[400], fs=fs) #highpass
    butter = Butter(fpass=[500, 700], fstop=[400, 800], fs=fs) #bandpass
    #butter = Butter(fpass=[400, 800], fstop=[500, 700], fs=fs) #bandstop

    #cheby1 = Cheby1(fpass=[400], fstop=[500], fs=fs) #lowpass
    #cheby1 = Cheby1(fpass=[500], fstop=[400], fs=fs) #highpass
    #cheby1 = Cheby1(fpass=[500, 700], fstop=[400, 800], fs=fs) #bandpass
    #cheby1 = Cheby1(fpass=[400, 800], fstop=[500, 700], fs=fs) #bandstop

    #cheby2 = Cheby2(fpass=[400], fstop=[500], fs=fs) #lowpass
    #cheby2 = Cheby2(fpass=[500], fstop=[400], fs=fs) #highpass
    #cheby2 = Cheby2(fpass=[500, 700], fstop=[400, 800], fs=fs) #bandpass
    #cheby2 = Cheby2(fpass=[400, 800], fstop=[500, 700], fs=fs) #bandstop

    #ellip = Ellip(fpass=[400], fstop=[500], fs=fs) #lowpass
    #ellip = Ellip(fpass=[500], fstop=[400], fs=fs) #highpass
    #ellip = Ellip(fpass=[500, 700], fstop=[400, 800], fs=fs) #bandpass
    #ellip = Ellip(fpass=[400, 800], fstop=[500, 700], fs=fs) #bandstop

