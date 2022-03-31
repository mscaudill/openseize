"""

"""

import numpy as np
import scipy.signal as sps

from openseize.filtering.bases import FIR

class Kaiser(FIR):
    """A callable Type I FIR Filter using a Kaiser window. 

    
    """

    def __init__(self, fpass, fstop, fs, gpass=1.0, gstop=40, **kwargs):
        """Initialize this Kaiser FIR filter."""

        super().__init__(fpass, fstop, gpass, gstop, fs, **kwargs)

    @property
    def num_taps(self):
        """ """

        ripple = abs(20 * np.log10(max(gpass, gstop)))
        ntaps, beta = sps.kaiserord(ripple, self.width)

    def cutoffs(self):
        """Returns cutoff frequencies for each transtion in this FIR.

        MOVE TO BASE
        
        DOCS
        """

        btype = self.btype()
        if btype in ('lowpass', 'highpass'):
            cutoffs = abs(self.fstop - self.fpass) / 2.0 + 
                      min(np.concatenate((self.fstop, self.fpass))
        elif btype in ('bandpass', 'bandstop'):
            cutoffs = [abs(self.fstop[0] - self.fpass[0]) / 2 + 
                       min(self.fstop[0], self.fpass[0]),
                       abs(self.fstop[1] - self.fpass[1]) / 2 + 
                       min(self.fstop[1], self.fpass[1])]
        
        return np.array(cutoffs)

    def _build(self):
        """ """

       

