import numpy as np
from scipy import signal

from openseize.io import edf, annotations
from openseize.io.dialogs import matched

# FIXME It is important that we add support for producers and iterative
# computation of both the autocovariances and their bases from a large
# number of training files. It would be good to handle ~ 10k training
# samples
#
# TODO I also want to figure out terminology that I can consistently use
# when developing models and be looking for ways to abstract 
#
class AutoCovariance:
    """ """

    def __init__(self):
        """ """

        pass

    def estimate(self, data, size, axis=-1):
        """Estimates an orthogonal basis for the autocovariances of True
        signals in data.

        Args:
            data: 2 or 3-D array
                An ndarray of signals used to build the basis. The 0th axis
                shape must match the number of signals used to build the
                basis.
            size: int
                The number of basis signals to keep.
            axis: int
                The sample axis of signals. If signals is 2-D this must be
                last axis.

        Returns:
        """

        # shuffle last two axes if axis is not last
        axis = np.arange(data.ndim)[axis]
        if axis < data.ndim:
            arr = np.moveaxis(data, -1, 1)

        # compute autocovariance
        data -= np.mean(data, axis=-1, keepdims=True)
        data /= np.std(data, axis=-1, keepdims=True)
        # flip and convolve to compute correlation
        reverse = np.flip(data, axis=axis)
        acv = signal.fftconvolve(data, reverse, axes=axis)
        # keep only positive lags
        acv = np.split(acv, [data.shape[-1] - 1], axis=-1)[1] 
        # swap the events and chs axis -- we find a basis for each ch
        acv = np.moveaxis(acv, 0, 1)
        
        # Build basis and store size num of basis vectors
        _, sigma, v = np.linalg.svd(acv, full_matrices=False)
        sigma = np.expand_dims(sigma[:size], axis=-1)
        sigma = sigma[:, :size, :]
        v = v[:, :size, :]
        # return a basis with only positive singular values
        self.sigmas = np.abs(sigma)
        self.basis = np.sign(sigma) * v
        return self.sigmas, self.basis

    def fit(self, data, labels, axis=-1):
        """ """

        # 2. build a logistic classifier for each channel
        # 3. train each classifier and store to classifiers attr
        
        # compute autocovariance
        data -= np.mean(data, axis=-1, keepdims=True)
        data /= np.std(data, axis=-1, keepdims=True)
        # flip and convolve to compute correlation
        reverse = np.flip(data, axis=axis)
        acv = signal.fftconvolve(data, reverse, axes=axis)
        # keep only positive lags
        acv = np.split(acv, [data.shape[-1] - 1], axis=-1)[1] 
        # swap the events and chs axis -- we find a basis for each ch
        acv = np.moveaxis(acv, 0, 1)

        # project autocovariances onto basis to get features
        # acvs @ basis == (chs, events, samples) @ (chs, samples, k)
        basis = np.moveaxis(self.basis, 1, -1)
        features = acv @ basis
        
        # build and train logistic
        classifiers = []
        for idx, f in enumerate(features):
            print('Training Model on Channel {}'.format(idx))
            model = LogisticRegression(penalty='none', random_state=0,
                                   class_weight='balanced')
        model.fit(f, labels)
        classifiers.append(model)



    def predict(self, eegs, times, **kwargs):
        """ """

        # validate that chs match the fitted channels?
        #
        pass

if __name__ == '__main__':

    DATA_DIR = '/media/matt/Zeus/swd/'
    
    model = AutoCovariance()

    def from_annotations(channels, lag=2, fs=5000):
        """Helper that constructs training data for a model from a selection
        of annotations."""

        # open dialog to fetch files
        paths = matched(titles=['Select EEGs', 'Select Annotations'], 
                        initialdir=DATA_DIR)

        
        # create a dict of eeg readers and sample locations
        data = dict()
        for epath, apath in paths:
            eeg = edf.Reader(epath)
            annotes = annotations.XueMat(apath).read()
            locs = [int(ann.time * fs) for ann in annotes]
            data[epath.stem] = {'reader' : eeg, 'locs' : locs}
    
        
        result = [] 
        for subdict in data.values():
            eeg, locs = subdict['reader'], subdict['locs']
            # expand locs to lag samples centered on locs
            samples = np.array([[-lag, lag]]) * fs // 2 + np.array([locs]).T
            for start, stop in samples:
                result.append(eeg.read(start, stop, channels))
        return np.array(result)
 

    data = from_annotations([0,1,2])
    sigmas, basis  = model.estimate(data, size=6)
    features = model.fit(data, labels=None)
