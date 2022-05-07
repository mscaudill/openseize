import numpy as np
from scipy import signal

from openseize.io import edf, annotations
from openseize.io.dialogs import matched, standard

from sklearn.linear_model import LogisticRegression

class AutoCovariance:
    """ """

    # The memory consuption is quite high due to the large fs (5kHz). This
    # limits the number of events we can use to build a basis and compute
    # features for training the logistic model. Since SWDs have a waveform
    # of duration 125 ms if we downsample the data to 500 Hz, we still have
    # 40 points to represent this waveform. This will allow us to store
    #    ~ 166K events for training with on 8 Gb of memory. 
    # PLAN:
    # 1. build downsampling
    # 2. downsample data and restest this model
    # 3. refactor and clean up the autocovariance and estimate methods
    #    assuming they can directly work on arrays of downsampled events
    # 4. work out if features need to be computed iteratively since this is
    #    currently a large 3-D matrix multiplication
    # 5. refactor predict so it can predict from discrete events or from
    #    an edf producer producing downsampled lag windows of data
    # 6. Test this detector in the continuous prediction mode
    # 7. Build methods to estimate the number of basis needed using scree
    #    plots
    # 8. build testing and demo files
    # 9. move the main functions for fetching events somewhere else for
    #    safe keeping
    # 10. make this model an inheritor of a model class for consistent
    #     building of detectors

    def __init__(self):
        """ """

        pass

    def _autocovariance(self, data):
        """ """

        # compute autocovariance
        data -= np.mean(data, axis=-1, keepdims=True)
        data /= np.std(data, axis=-1, keepdims=True)
        # flip and convolve to compute correlation
        reverse = np.flip(data, axis=-1)
        acv = signal.fftconvolve(data, reverse, axes=-1)
        # keep only positive lags
        acv = np.split(acv, [data.shape[-1] - 1], axis=-1)[1] 
        # swap the events and chs axis -- we find a basis for each ch
        acv = np.moveaxis(acv, 0, 1)
        return acv

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

        print('building a basis from {} True events'.format(data.shape[0]))
        arr = data
        # shuffle last two axes if axis is not last
        axis = np.arange(data.ndim)[axis]
        if axis < data.ndim-1:
            arr = np.moveaxis(data, -1, 1)

        acv = self._autocovariance(arr)

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

        #handle axis if not last

        acv = self._autocovariance(data)
        # project autocovariances onto basis to get features
        # acvs @ basis == (chs, events, samples) @ (chs, samples, k)
        basis = np.moveaxis(self.basis, 1, -1)

        # This is costly in terms of memory -- could be iterative
        features = acv @ basis
        
        # build and train logistic
        # Place events on 0th
        features = np.swapaxes(features, 0,1)
        #flatten to events x (channels * features)
        x = features.reshape(features.shape[0], -1)
        print('building logisitic model from {} events'.format(x.shape[0]))
        logmodel = LogisticRegression(penalty='none', random_state=0,
                                      class_weight='balanced')
        logmodel.fit(x, labels)
        self._model = logmodel
        return self._model

    def predict(self, data, continuous=False, axis=-1):
        """ """

        # 1. build autocovariance
        # 2. compute features by projection and flatten
        # 3. call _models' predict method and return

        acv = self._autocovariance(data)
        basis = np.moveaxis(self.basis, 1, -1)
        x = acv @ basis
        x = np.swapaxes(x, 0, 1)
        x = np.reshape(x, (x.shape[0], -1))
        return self._model.predict_proba(x)

if __name__ == '__main__':

    from pathlib import Path
    from siftnets.archive.archive import Archive 
    
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

    def from_archive(channels, lag=2, fs=5000, reduction=[1, 1], seed=0):
        """Helper that constructs a training data set from an archive with
        both real (True) and artifact (False) swd events."""

        # open dialog to fetch files
        paths = standard('askopenfilenames', title='Select EEGs', 
                         initialdir=DATA_DIR)

        arc = Archive.load(
                Path(DATA_DIR).joinpath('arc22_2021_04_19_09h53m.pkl'))

        # We have to build labels because the arc does not store them by
        # path. This will be fixed
        cnts = [arr.shape[0] for arr in arc.epochs.values()]
        csums = np.cumsum(np.insert(cnts, 0, 0))
        slices = np.array(list(zip(csums, csums[1:])))
        slices = dict(zip(arc.epochs.keys(), slices))
        labels = {name: arc.labels[slice(*edges)] for name, edges in
                slices.items()}

        data = dict()
        for path in paths:
            eeg = edf.Reader(path)
            epochs = arc.epochs[path.stem] 
            labs = labels[path.stem] 
            locs = np.mean(epochs, axis=1).astype(int)
            data[path.stem] = {'reader' : eeg, 'locs' : locs, 'labels': labs}

        # The archive has ~ 40 times the number of False as True SWDs
        # to conserve memory we should cut this down so we build a better
        # logistic model. Ideally clients will supply hand marked SWDs and
        # artifacts for training this model.
        if (np.array(reduction) > 1).any():
            reduced = dict()
            rng = np.random.default_rng(seed)
            for name, subdict in data.items():
                eeg = subdict['reader']
                labs = subdict['labels']
                locs = subdict['locs']
                # get True and False indices
                pos, negs = np.nonzero(labs)[0], np.nonzero(~labs)[0]
                # Compute reduced sizes and choose
                sizes = np.array([len(pos), len(negs)]) / reduction
                sizes = sizes.astype(int)
                pos = rng.choice(pos, size=sizes[0], replace=False,
                                 shuffle=False)
                negs = rng.choice(negs, size=sizes[1], replace=False,
                                  shuffle=False)
                indices = np.sort(np.concatenate((pos, negs)))
                red_locs = locs[indices]
                red_labels = labs[indices]
                data[name].update({'locs': red_locs, 'labels': red_labels})

        result = [], []
        for subdict in data.values():
            eeg, locs = subdict['reader'], subdict['locs']
            # store labels to results
            result[1].extend(subdict['labels'])
            # expand locs to lag samples centered on locs
            samples = np.array([[-lag, lag]]) * fs // 2 + np.array([locs]).T
            for start, stop in samples:
                result[0].append(eeg.read(start, stop, channels))
        return np.array(result[0]), np.array(result[1])





    
    # Estimate the basis
    data = from_annotations([0,1,2])

    """
    sigmas, basis  = model.estimate(data, size=6)

    # Train logistic classifiers
    data, labels = from_archive(channels=[0,1,2], reduction=[1, 20])
    clfs = model.fit(data, labels)

    # Make Predictions
    data, labels = from_archive(channels=[0,1,2], reduction=[1,20])
    predictions = model.predict(data)

    # codify this
    x = predictions[:,1] > 0.25
    comparison =np.stack((x, labels), axis=1)
    cnt = len(labels)
    pos, neg = np.count_nonzero(labels), np.count_nonzero(~labels)
    tp = np.count_nonzero(np.all(comparison, axis=1))
    tn = np.count_nonzero(np.all(~comparison, axis=1))
    fp = np.count_nonzero(x) - tp
    fn = pos + neg - tp - fp - tn

    tpr = tp / pos
    fpr = fp / neg

    print('pos = {}'.format(pos))
    print('neg = {}'.format(neg))
    print('total = []'.format(cnt))
    print('tp = {}'.format(tp))
    print('fp = {}'.format(fp))
    print('tn = {}'.format(tn))
    print('fn = {}'.format(fn))
    print('tpr = {}'.format(tpr))
    print('fpr = {}'.format(fpr))

    print('ACC = {}'.format((tp + tn) / (cnt)))
    """
