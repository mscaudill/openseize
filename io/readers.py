import numpy as np

from openseize.io import headers

class EDFReader:
    """A European Data Format reader supporting the reading of both EDF and
    EDF+ formats.

    The EDF specification has a header section followed by data records
    Each data record contains all signals stored sequentially. EDF+
    files include an annotation signal within each data record. To
    distinguish these signals we refer to data containing signals as
    channels and annotation signals as annotation. For details on the EDF/+
    file specification please see:

    https://www.edfplus.info/specs/index.html

    Currently, this reader does not support the reading of annotation
    signals.
    """

    def __init__(self, path):
        """Initialize this EDFReader with a path and construct header."""

        self.path = path
        self.header = headers.EDFHeader(path)

    def __getattr__(self, name):
        """On attribute fetch failure attempt lookup on header."""

        return self.header[name]

    @property
    def annotated(self):
        """Returns True if this is EDF contains annotations."""

        return True if 'EDF Annotations' in self.names else False
        
    @property
    def annotation(self):
        """Returns annotations signal index if present & None otherwise."""
        
        result = None
        if self.annotated:
            result = self.names.index('EDF Annotations') 
        return result

    @property
    def channels(self):
        """Returns the 'ordinary signal' indices."""

        signals = list(range(self.num_signals))
        if self.annotated:
            signals.pop(self.annotation)
        return signals
    
    @property
    def record_map(self):
        """Returns a list of slice objects for each signal in a record."""

        scnts = np.insert(self.samples_per_record,0,0)
        cum = np.cumsum(scnts)
        return list(slice(a, b) for (a, b) in zip(cum, cum[1:]))

    @property
    def lengths(self):
        """Returns summed sample count across records for each channel.

        The last record of the EDF may not be completely filled with
        recorded signal values. lengths measures the number of recorded
        values by subtracting off the appended 0s on the last record.
        """

        #get & read the last record
        last_rec = (self.num_records-1, self.num_records)
        last_sample = np.array(self.samples_per_record) * self.num_records
        arr = self.records(*last_rec)
        arr = arr.reshape(1, sum(self.samples_per_record))
        #subtract the appends (usually 0s) from each channels len
        for ch in self.channels:
            ch_data = arr[:, self.record_map[ch]].flatten()
            last_sample[ch] -= len(np.where(ch_data == 0)[0])
        return np.array([last_sample[ch] for ch in self.channels])

    def transform(self, arr, axis=-1):
        """Linearly transforms a 2-D integer array fetched from this EDF.

        The physical values p are linearly mapped from the digital values d:
        p = slope * d + offset, where the slope = 
        (pmax -pmin) / (dmax - dmin) & offset = p - slope * d for any (p,d)

        Returns: ndarray with shape matching input shape & float64 dtype
        """

        pmaxes = np.array(self.physical_max)
        pmins = np.array(self.physical_min)
        dmaxes = np.array(self.digital_max)
        dmins = np.array(self.digital_min)
        slopes = (pmaxes - pmins) / (dmaxes - dmins)
        offsets = (pmins - slopes * dmins)
        #expand to 2-D for broadcasting
        slopes = np.expand_dims(slopes[self.channels], axis=axis)
        offsets = np.expand_dims(offsets[self.channels], axis=axis)
        result = arr * slopes
        result += offsets
        return result

    def find_records(self, start, stop):
        """Returns tuples (one per signal) of start, stop record numbers
        that include the start, stop sample numbers

        Args:
            start (int):                start of sample range to read
            stop (int):                 stop of sample range to read
        """

        spr = np.array(self.samples_per_record)[self.channels]
        starts = start // spr
        stops = np.ceil(stop / spr).astype('int')
        #ensure stops do not exceed filled records
        max_records = self.lengths // spr
        stops = [min(a, b) for a, b in zip(stops, max_records)]
        return list(zip(starts, stops))

    def records(self, a, b):
        """Reads all samples from the ath to bth record.

        Returns: a 1D array of length (b-a) * sum(samples_per_record)
        """

        cnt = b - a
        #each sample is represented as a 2-byte integer
        bytes_per_record = sum(self.samples_per_record) * 2
        #return offset in bytes & num samples spanning a to b
        offset = self.header_bytes + a * bytes_per_record
        nsamples = cnt * sum(self.samples_per_record)
        with open(self.path, 'rb') as f:
            arr = np.fromfile(f, dtype='<i2', count=nsamples, offset=offset)
        return arr

    def read(self, start, stop):
        """Returns samples from start to stop for all channels of this EDF.

        Args:
            start (int):            start sample to begin reading
            stop (int):             stop sample to end reading (exclusive)

        Returns: array of shape chs x samples with float64 dtype
        """

        if start > max(self.lengths):
            msg = 'start idx {} exceeds number of samples {}'
            raise IndexError(msg.format(start, self.lengths))
        #locate record ranges for chs & find unique ones to read
        recs = self.find_records(start, stop)
        urecs = set(recs)
        #read each unique record range
        reads = {urec: self.records(*urec) for urec in urecs}
        #reshape each read to num_records x summed samples per rec
        reads = {urec: arr.reshape(-1, sum(self.samples_per_record)) 
                 for urec, arr in reads.items()}
        #perform final slicing and transform for each channel
        result = []
        for idx, (channel, rec) in enumerate(zip(self.channels, recs)):
            #get preread array and slice out channel
            arr = reads[rec]
            arr = arr[:, self.record_map[channel]].flatten()
            #adjust start & stop relative to first read record & store
            a = start - rec[0] * self.samples_per_record[channel]
            b = a + (stop - start)
            result.append(arr[a:b])
        result = np.stack(result, axis=0)
        #transform & return
        result = self.transform(result)
        return result




if __name__ == '__main__':

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    reader = EDFReader(path)





        
