import numpy as np

from scipy.stats import chi2

from openseize.core import numerical as nm
from openseize import producer


class PSD:
    """

    """

    def __init__(self, pro, fs):
        """Initialize this Welch."""

        self.pro = pro
        self.fs = fs
    
        
    def estimate(self, resolution=0.5, window='hann', overlap=0.5, axis=-1,
                 detrend='constant', scaling='density'):
        """

        """

        nfft = int(self.fs / resolution)
        freqs, psd_pro = nm.welch(self.pro, self.fs, nfft, window, overlap,
                                  axis, detrend, scaling)

        self.freqs, self.psd_pro = freqs, psd_pro

        result = 0
        for cnt, arr in enumerate(self.psd_pro, 1):
            result = result + 1 / cnt * (arr - result)
        self.avg_psd = result
        
        return freqs, result

    def confidence_interval(self, alpha=0.05):
        """ """

        pass

    def plot(self):
        pass


if __name__ == '__main__':

    from openseize.io import edf
    from openseize.demos import paths


    def fetch_data(start, stop):
        """ """

        fp = paths.locate('recording_001.edf')
        with edf.Reader(fp) as reader:
            arr = reader.read(start, stop)

        return arr
    
    data = fetch_data(0, 127000)
    pro = producer(data, chunksize=20000, axis=-1)

    estimator = PSD(pro, fs=5000)
    freqs, avg_psd = estimator.estimate()

