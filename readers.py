import numpy as np

from openseize import headers

class EDFReader:
    """A European Data Format reader supporting the reading of both EDF and
    EDF+ formats.

    The EDF specification has a header section followed by a data records
    section. Each data record contains all signals stored sequentially. EDF+
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

    @property
    def num_samples(self, filledvalued=0):
        """Returns total sample count across records for each channel."""
        #FIXME This seems a little hackish

        last_record = (self.header.num_records-1, self.header.num_records)
        # get bytevector for last record
        vec = self.records_to_bytes(*last_record)
        print(vec)
        with open(self.path, 'rb') as f:
            data = np.fromfile(f, dtype='<i2', count=vec[1], offset=vec[0])
        data = data.reshape(-1, sum(self.sample_counts))
        lengths = []
        for sig in range(self.header.num_signals):
            d=np.count_nonzero(data[:, self.record_map[sig]].flatten())
            lengths.append(d)
        return self.sample_counts * (self.header.num_records-1) + lengths

    def transform(self, arr, axis=-1):
        """Linearly transforms a 2-D integer array fetched from this EDF.

        The physical values p are linearly mapped from the digital values d:
        p = slope * d + offset, where the slope = 
        (pmax -pmin) / (dmax - dmin) & offset = p - slope * d for any (p,d)

        Returns: ndarray with shape matching input shape & float64 dtype
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
        """Returns tuples (one per signal) of start, stop record numbers
        that include the start, stop sample numbers

        Args:
            start (int):                start of sample range to read
            stop (int):                 stop of sample range to read
        """

        starts = start // self.sample_counts
        stops = np.ceil(stop / self.sample_counts).astype('int')
        return list(zip(starts, stops))

    def records_to_bytes(self, start, stop):
        """Converts a range of record numbers to a tuple of byte-offset and
        number of samples. This tuple is termed a 'bytevector'.

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
        """Returns samples from start to stop for all channels of this EDF.

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
            reads = {uvec: np.fromfile(f, dtype='<i2', count=uvec[1], 
                     offset=uvec[0]) for uvec in uniqvecs}
        #reshape each read to num_records x summed sample_counts
        reads = {uvec: arr.reshape(-1, sum(self.sample_counts)) 
                 for uvec, arr in reads.items()}
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





        
