"""A collection of callable Finite Impulse Response filters.

This collection includes the Kaiser parametric filter and non-parametric 
general cosine window filters including. Each filter inherits the ability 
to plot itself and can be called on an ndarray or a producer of ndarrays.
Please see the FIR base class in openseize.filtering.bases for inherited
properties and methods.

The Kaiser windows can be designed to meet the stricter of the pass and stop
band attenuation criteria within the specified transition width. The general
cosine windows (GCW) FIRS (Rectangular, Hanning, Hamming, Bartlett & 
Blackman) do not allow for tuning of the attenuation as these are inherent 
to the window shape. Thus to use the GCWs it is necessary to pick a specific
GCW that meets the peak error spec. These are provided in the documentation
for each windowed FIR. As such the Kaiser FIR is the simpler choice for
general filtering unless some specific need (such as fast roll-offs) is 
required.

Typical usage example:
    
    # build a lowpass Kaiser FIR with and edge frequencies of 500 and 600 Hz
    # that has a maximum passband loss of 1 dB ~ 11 % & a minimum stop band
    # attenuation of 40 dB ~ 99% amplitude attenuation.
    kfir = Kaiser(fpass=500, fstop=600, fs=5000, gpass=1.0, gstop=40)

    # plot this filter's impulse and frequency response
    kfir.plot()

    # finally apply the filter to a producer of ndarrays called 'data'.
    # and account for the group delay of the filter with mode='same'
    filtered = kfir(data, chunksize=10e6, axis=-1, mode='same')

    # process each filtered chunk
    for arr in filtered:
        # do something with filtered chunk 
"""

import numpy as np
import scipy.signal as sps

from openseize.filtering.bases import FIR

class Kaiser(FIR):
    """A callable Type I FIR Filter using a Kaiser window. 

    Designs a Kaiser windowed filter that meets the stricter of the pass and
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

    References:
        1. Ifeachor E.C. and Jervis, B.W. (2002). Digital Signal Processing:
           A Practical Approach. Prentice Hall
        2. Oppenheim, Schafer, "Discrete-Time Signal Processing".
    """

    def __init__(self, fpass, fstop, fs, gpass=1.0, gstop=40, **kwargs):
        """Initialize this Kaiser windowed FIR filter."""

        super().__init__(fpass, fstop, gpass, gstop, fs, **kwargs)

    @property
    def numtaps(self):
        """Returns the number of taps needed to meet the stricter of the
        pass and stop band criteria and the kaiser window parameter beta.

        The gpass attr of this FIR is the max loss in the passband. This is
        not the same as the attenuation in the pass band so we first compute
        the attenuation in the passband and compare against the attenuation
        in the stop band. The largest attenuation is then used to compute
        the minimum number of taps.

        References:
            Oppenheim, Schafer, "Discrete-Time Signal Processing", 
            pp.475-476.
        """
        
        ripple = max(self.pass_attenuation, self.gstop)
        ntaps, _ = sps.kaiserord(ripple, self.width/self.nyq)
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps

    @property
    def window_params(self):
        """Returns the additional beta paramater required to create this 
        FIRs Kaiser window."""
        
        ripple = max(self.pass_attenuation, self.gstop)
        return [sps.kaiser_beta(ripple)]


class Rectangular(FIR):
    """A callable type I FIR using a rectangular window.

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

    Characteristics:
        main lobe width (MLW) = 4 pi / len(taps) 
        side lobe height (SLH) = -13.3 dB
        side lobe roll-off rate (SLRR) = -6 dB/octave
        approximate peak error (APE) = -21 dB
        
    It has large spectral leakage and its primary use case is for working with
    periodic signals whose frequencies are multiples of the window length.
    """

    def __init__(self, fpass, fstop, fs):
        """Initialize this Rectangular Windowed FIR."""
        
        # for plotting, provide a gpass calculated from the peak error
        peak_err = -21
        gpass = -20 * np.log10(1 - 10 ** (peak_err / 20))
        super().__init__(fpass, fstop, gpass=gpass, gstop=peak_err, fs=fs)

    @property
    def numtaps(self):
        """Return the number of taps to meet the transition width."""
        
        ntaps = int(4 / (self.width / self.nyq))
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps


class Hanning(FIR):
    """A callable type I FIR using a Hann window.

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

    Characteristics:
        main lobe width (MLW) = 8 pi / len(taps) 
        side lobe height (SLH) = -31.5 dB
        side lobe roll-off rate (SLRR) = -18 dB/octave
        approximate peak error (APE) = -44 dB
        
    The Hann window is a good general purpose window with reduced spectral
    leakage.
    """

    def __init__(self, fpass, fstop, fs):
        """Initialize this Rectangular Windowed FIR."""
        
        # for plotting, provide a gpass calculated from the peak error
        peak_err = -44
        gpass = -20 * np.log10(1 - 10 ** (peak_err / 20))
        super().__init__(fpass, fstop, gpass=gpass, gstop=peak_err, fs=fs)

    @property
    def numtaps(self):
        """Return the number of taps to meet the transition width."""
        
        ntaps = int(8 / (self.width / self.nyq))
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps
  

class Hamming(FIR):
    """A callable type I FIR using a Hamming window.

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

    Characteristics:
        main lobe width (MLW) = 8 pi / len(taps) 
        side lobe height (SLH) = -43.8 dB
        side lobe roll-off rate (SLRR) = -6 dB/octave
        approximate peak error (APE) = -53 dB
        
    The Hamming window is a good general purpose window with strong
    attenuation in the pass and stop bands.
    """

    def __init__(self, fpass, fstop, fs):
        """Initialize this Rectangular Windowed FIR."""
        
        # for plotting, provide a gpass calculated from the peak error
        peak_err = -53
        gpass = -20 * np.log10(1 - 10 ** (peak_err / 20))
        super().__init__(fpass, fstop, gpass=gpass, gstop=peak_err, fs=fs)

    @property
    def numtaps(self):
        """Return the number of taps to meet the transition width."""
        
        ntaps = int(8 / (self.width / self.nyq))
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps


class Bartlett(FIR):
    """A callable type I FIR using a Bartlett (triangular) window.

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

    Characteristics:
        main lobe width (MLW) = 8 pi / len(taps) 
        side lobe height (SLH) = -26.5 dB
        side lobe roll-off rate (SLRR) = -12 dB/octave
        approximate peak error (APE) = -25 dB
        
    The Bartlett window has a narrow main lobe but higher side lobes and
    thus leakage.
    """

    def __init__(self, fpass, fstop, fs):
        """Initialize this Rectangular Windowed FIR."""
        
        # for plotting, provide a gpass calculated from the peak error
        peak_err = -25
        gpass = -20 * np.log10(1 - 10 ** (peak_err / 20))
        super().__init__(fpass, fstop, gpass=gpass, gstop=peak_err, fs=fs)

    @property
    def numtaps(self):
        """Return the number of taps to meet the transition width."""
        
        ntaps = int(8 / (self.width / self.nyq))
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps


class Blackman(FIR):
    """A callable type I FIR using a Blackman window.

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

    Characteristics:
        main lobe width (MLW) = 12 pi / len(taps) 
        side lobe height (SLH) = -58.2 dB
        side lobe roll-off rate (SLRR) = -18 dB/octave
        approximate peak error (APE) = -74 dB
        
    The Blackman window has a wider main lobe but greater attenuation in the
    pass and stop bands with a good roll-off.
    """

    def __init__(self, fpass, fstop, fs):
        """Initialize this Rectangular Windowed FIR."""
        
        # for plotting, provide a gpass calculated from the peak error
        peak_err = -74
        gpass = -20 * np.log10(1 - 10 ** (peak_err / 20))
        super().__init__(fpass, fstop, gpass=gpass, gstop=peak_err, fs=fs)

    @property
    def numtaps(self):
        """Return the number of taps to meet the transition width."""
        
        ntaps = int(12 / (self.width / self.nyq))
        # odd tap number to ensure group delay is integer samples
        return ntaps + 1 if ntaps % 2 == 0 else ntaps



if __name__ == '__main__':
   
    import matplotlib.pyplot as plt

    fpass = [400]
    fstop = [500]

    #fpass = [500]
    #fstop = [400]
    
    #fpass = [400, 800]
    #fstop = [300, 900]
    
    #fpass = [400, 900]
    #fstop = [450, 850]
    
    kfir = Kaiser(fpass=fpass, fstop=fstop, gpass=1, gstop=40, fs=5000)
    kfir.plot(worN=2048)

    filt = Hanning(fpass=fpass, fstop=fstop, fs=5000)

    """
    w, h = sps.freqz(kfir.coeffs, fs=5000, worN=2000)
    fig, axarr = plt.subplots(2,1)
    axarr[0].plot(w, 20*np.log10(np.abs(h)))
    axarr[1].plot(w, np.abs(h))
    [ax.grid(alpha=0.5) for ax in axarr]
    [ax.axvline(fp, color='red') for fp in fpass for ax in axarr]
    [ax.axvline(fs, color='red') for fs in fstop for ax in axarr]
    [axarr[0].axvline([c], color='pink') for c in kfir.cutoff]
    [axarr[1].axvline([c], color='pink') for c in kfir.cutoff]
    plt.ion()
    plt.show()
    """




       
