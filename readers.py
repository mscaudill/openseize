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
    def rates(self):
        """Returns the sampling rates of the channels."""

        res = self.header.samples_per_record / self.header.record_duration
        return [res[ch] for ch in self.channels]

    def transform(self, arr):
        """Linearly transforms an integer array fetched from this EDF.

        The physical values p are linearly mapped from the digital values d:
        p = slope * d + offset, where the slope = 
        (pmax -pmin) / (dmax - dmin) & offset = p - slope * d for any (p,d)
        """
        #FIXME I may need to operate in-place

        pmaxes = np.array(self.header.physical_max)
        pmins = np.array(self.header.physical_min)
        dmaxes = np.array(self.header.digital_max)
        dmins = np.array(self.header.digital_min)
        slopes = ((pmaxes - pmins) / (dmaxes - dmins))[self.channels]
        offsets = (pmins - slopes * dmins)[self.channels]
        #apply to input arr along axis
        return arr * slopes + offsets
    
    def records(self, start, stop):
        """Returns tuples of start, stop records for each channel that
        includes the start, stop samples.

        Args:
            start (int):                start of sample range to read
            stop (int):                 stop of sample range to read
        """

        spr = np.array(self.header.samples_per_record)[self.channels]
        starts = start // spr
        stops = np.ceil(stop / spr).astype('int')
        return list(zip(starts, stops))

    def bytevector(self, start_rec, stop_rec):
        """Returns a tuple with byte offset and num of samples to read that
        include the start and stop records.

        Args:
            start_rec (int):            start record number 
            stop_rec (int):             stop record number
        """

        nrecords = stop_rec - start_rec
        #get the start of the data section of this EDF
        data_start = self.header.header_bytes
        #get the number of bytes in a record (2 bytes per sample)
        bytes_per_record = sum(self.header.samples_per_record) * 2
        #return byte offset and samples to read
        offset = data_start + start_rec * bytes_per_record
        nsamples = nrecords * sum(self.header.samples_per_record)
        return offset, nsamples

    def reshaper(self, flattened):
        """ """

        num_recs = len(flattened) // sum(self.header.samples_per_record)
        num_chs = len(self.channels)
        arr = flattened.reshape(num_recs,
                                sum(self.header.samples_per_record))
        #slice out the annotations
        arr = arr[:, self.channels]
        #FIXME BELOW I am assuming equal samples per channel!
        #reshape to num_recs x num_chs x samples_per_channel
        arr = arr.reshape(num_recs, num_chs, -1)

    def read(self, start, stop):
        """ """

        #get records and bytevectors for each channel
        recs = self.records(start, stop)
        bytevecs = [self.bytevector(*rec) for rec in recs]
        #get the unique bytevectors we need to read
        uvectors = set(bytevecs) 
        #perform the minimum number of reads
        with open(self.path, 'rb') as f:
            reads = {uvec: np.fromfile(f, dtype='<i2', count=uvec[1], 
                     offset=uvec[0]) for uvec in uvectors}

        return reads


if __name__ == '__main__':

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    reader = EDF(path)





        
