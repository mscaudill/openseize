import numpy as np

from openseize import headers

class EDFReader:
    """

    """

    #signals will be a list of all signals in edf
    #channels will be a list of all ordinary signals in edf
    #annotations will be the annotation signal in edf (may be None)

    def __init__(self, path):
        """Initialize this EDFReader with a path and construct header."""

        self.path = path
        self.header = headers.EDFHeader(path)

    @property
    def annotated(self):
        """Returns True if this is EDF contains annotations."""

        return True if 'EDF Annotations' in self.header.names else False
        
    @property
    def annotation(self):
        """Returns annotations signal index if present & None otherwise."""
        
        result = None
        if self.annotated:
            result = self.header.names.index('EDF Annotations') 
        return result

    @property
    def channels(self):
        """Returns the 'ordinary signal' indices."""

        signals = list(range(self.header.num_signals))
        if self.annotated:
            signals.pop(self.annotation)
        return signals

    @property
    def sample_counts(self):
        """Returns the number of samples for each signal in a record."""

        return np.array(self.header.samples_per_record)

    @property
    def record_map(self):
        """Returns a list slice objects for each signal in a record."""

        scnts = np.insert(self.sample_counts,0,0)
        cum = np.cumsum(scnts)
        return list(slice(a, b) for (a, b) in zip(cum, cum[1:]))

    def transform(self, arr, axis=-1):
        """Linearly transforms a 2-D integer array fetched from this EDF.

        The physical values p are linearly mapped from the digital values d:
        p = slope * d + offset, where the slope = 
        (pmax -pmin) / (dmax - dmin) & offset = p - slope * d for any (p,d)

        Returns: an array with shape matching the input shape & dtype of
                 float64
        """

        pmaxes = np.array(self.header.physical_max)
        pmins = np.array(self.header.physical_min)
        dmaxes = np.array(self.header.digital_max)
        dmins = np.array(self.header.digital_min)
        slopes = (pmaxes - pmins) / (dmaxes - dmins)
        offsets = (pmins - slopes * dmins)
        #expand to 2-D for broadcasting
        slopes = np.expand_dims(slopes[self.channels], axis=axis)
        offsets = np.expand_dims(offsets[self.channels], axis=axis)
        result = arr * slopes
        result += offsets
        return result
    
    def records(self, start, stop):
        """Returns a tuple of start, stop record numbers that include the
        start, stop sample numbers.

        Args:
            start (int):                start of sample range to read
            stop (int):                 stop of sample range to read
        """

        starts = start // self.sample_counts
        stops = np.ceil(stop / self.sample_counts).astype('int')
        return list(zip(starts, stops))

    def records_to_bytes(self, start, stop):
        """Converts a range of record numbers to a tuple of byte-offset and
        number of samples. This tuple is a 'bytevector'.

        Args:
            start (int):                start of record range to convert
            stop (int):                 stop of record range to convert

        Returns: a tuple of byte-offset and number of samples
        """

        cnt = stop - start
        #get start of records sec of this EDF and bytes per record
        records_start = self.header.header_bytes
        samples = sum(self.sample_counts)
        #each sample is represented as a 2-byte integer
        bytes_per_record = samples * 2
        #return offset in bytes & num samples spanning start to stop
        offset = records_start + start * bytes_per_record
        nsamples = cnt * samples
        return offset, nsamples

    def read(self, start, stop):
        """Returns samples between start & stop for all channels.

        Args:
            start (int):            start sample to begin reading
            stop (int):             stop sample to end reading (exclusive)

        Returns: array of shape chs x samples with float64 dtype
        """

        #get records and convert to bytevectors
        recs = np.array(self.records(start, stop))[self.channels]
        bytevectors = [self.records_to_bytes(*rec) for rec in recs]
        #get unique bytevectors & perform minimum reads
        uniqvecs = set(bytevectors) 
        with open(self.path, 'rb') as f:
            reads = {vec: np.fromfile(f, dtype='<i2', count=vec[1], 
                     offset=vec[0]) for vec in uniqvecs}
        #reshape reads to num_records x sample_counts
        reads = {k: arr.reshape(-1, sum(self.sample_counts)) for k, arr in
                reads.items()}
        #perform final slicing and transform for each channel
        result = np.zeros((len(self.channels), stop-start))
        for idx, (channel, rec, bytevec) in enumerate(zip(self.channels,
                                                  recs, bytevectors)):
            #get preread array and slice out channel
            arr = reads[bytevec]
            arr = arr[:, self.record_map[channel]].flatten()
            #adjust start & stop relative to first read record & store
            a = start - rec[0] * self.sample_counts[channel]
            b = a + (stop - start)
            result[idx] = arr[a:b]
        #transform & return
        result = self.transform(result)
        return result


if __name__ == '__main__':

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    newreader = EDFReader(path)





        
