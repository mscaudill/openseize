import numpy as np

from openseize import headers

class EDF:
    """

    """

    #signals will be a list of all signals in edf
    #channels will be a list of all ordinary signals in edf
    #annotations will be the annotation signal in edf (may be None)

    def __init__(self, path):
        """ """

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
    def record_map(self):
        """Returns a list slice objects for each signal in a record."""

        spr = self.header.samples_per_record.copy()
        spr.insert(0,0)
        cum_spr = np.cumsum(spr)
        return list(slice(a, b) for (a, b) in zip(cum_spr, cum_spr[1:]))

    def transform(self, arr, axis=-1):
        """Linearly transforms an integer array fetched from this EDF.

        The physical values p are linearly mapped from the digital values d:
        p = slope * d + offset, where the slope = 
        (pmax -pmin) / (dmax - dmin) & offset = p - slope * d for any (p,d)
        """
        #FIXME I may need to operate in-place

        pmaxes = np.array(self.header.physical_max)[self.channels]
        pmins = np.array(self.header.physical_min)[self.channels]
        dmaxes = np.array(self.header.digital_max)[self.channels]
        dmins = np.array(self.header.digital_min)[self.channels]
        slopes = (pmaxes - pmins) / (dmaxes - dmins)
        offsets = (pmins - slopes * dmins)
        #expand for array broadcast
        slopes = np.expand_dims(slopes, axis=axis)
        offsets = np.expand_dims(offsets, axis=axis)
        #apply to input arr along axis
        return arr * slopes + offsets
    
    def records(self, start, stop):
        """Returns a tuple of start, stop record numbers that include the
        start, stop sample numbers.

        Args:
            start (int):                start of sample range to read
            stop (int):                 stop of sample range to read
        """

        spr = np.array(self.header.samples_per_record)
        starts = start // spr
        stops = np.ceil(stop / spr).astype('int')
        return list(zip(starts, stops))

    def records_to_bytes(self, start, stop):
        """Converts a range of record numbers to a tuple of byte-offset and
        number of samples to read.

        Args:
            start (int):                start of record range to convert
            stop (int):                 stop of record range to convert
        """

        cnt = stop - start
        #get start of records sec of this EDF and bytes per record
        records_start = self.header.header_bytes
        samples_per_record = sum(self.header.samples_per_record)
        bytes_per_record = samples_per_record * 2
        #return offset in bytes & num samples spanning start to stop
        offset = records_start + start * bytes_per_record
        nsamples = cnt * samples_per_record
        return offset, nsamples

    def read(self, start, stop):
        """ """

        chs = self.channels
        #get records and bytevectors for each channel
        recs = np.array(self.records(start, stop))[chs]
        bytevectors = [self.records_to_bytes(*rec) for rec in recs]
        #get the unique bytevectors we need to read
        ubytes = set(bytevectors) 
        #perform the minimum number of reads
        with open(self.path, 'rb') as f:
            reads = {ubyte: np.fromfile(f, dtype='<i2', count=ubyte[1], 
                     offset=ubyte[0]) for ubyte in ubytes}
        #
        result = np.zeros((len(self.channels), stop-start))
        for idx, (ch, rec, bvec) in enumerate(zip(chs, recs, bytevectors)):
            #fetch the preread 1-D array of integers
            arr = reads[bvec]
            #reshape into num_records x total samples in record
            arr = arr.reshape(-1, sum(self.header.samples_per_record))
            #slice out channel using record_map and flatten
            arr = arr[:, self.record_map[ch]].flatten()
            #get start stop relative to record
            a = start - rec[0] * self.header.samples_per_record[ch]
            b = a + (stop - start)
            result[idx] = arr[a:b]
        #transform the result and return
        result = self.transform(result)
        return result


if __name__ == '__main__':

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    reader = EDF(path)





        
