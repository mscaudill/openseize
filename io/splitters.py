from pathlib import Path

from openseize.io import headers, readers, writers
from openseize.types import mixins

class EDF(mixins.ViewInstance):
    """A tool for creating multiple EDFs using a subset of the data from
    a single EDF file.

    EDFs may contain channels corresponding to more than one subject. This
    tool allows clients to split these EDFs into multiple EDFs one per 
    subject in the unsplit EDF.

    Note: Does not overwrite the original EDF.
    """

    def __init__(self, path):
        """Initialize with path to EDF to be split & read its header."""

        self.read_path = Path(path)
        with readers.EDF(self.read_path) as infile:
            self.header = infile.header

    def _channels(self, attr, values):
        """Returns channels of the header's attribute matching values."""

        return [getattr(self.header, attr).index(v) for v in values]

    def _createpath(self, idx):
        """Creates a default writepath from this Splitter's readpath by
        appending idx to the readpath stem.

        Returns: new write Path instance
        """
        
        p = self.read_path
        stem = p.stem + '_{}'.format(idx)
        return Path.joinpath(p.parent, stem).with_suffix(p.suffix)
        
    def _split(self, attr, values, writepath, **kwargs):
        """Writes a single EDF to writepath filtering by attr in header
        whose values match values.

        Args:
            attr (str):         named attr of filter splitter's header
                                (see io.headers.filter)
            values (list):      values to filter splitter's header
            writepath (Path):   write location of this split
            kwargs:             passed to writers write method

        Developer note: This method has been excised from the split method
        to allow for multiprocessing of file splitting in the future.
        """

        with readers.EDF(self.read_path) as reader:
            #get channel indices corresponding to values
            channels = self._channels(attr, values)
            with writers.EDF(writepath) as writer:
                writer.write(reader.header, reader, channels, **kwargs) 

    def split(self, by, groups, writepaths=None, **kwargs):
        """Creates multiple EDFs from the EDF at Splitter's readpath

        Args:
            by (str):           header field to split EDF data on (e.g.
                                channels or names or transducers etc.)
            groups (lists):     lists of list specifying values of the by
                                attribute to write to each split edf; e.g. 
                                if by='channels' and groups = [[1,3], [2,4]]
                                then channels [1,3] will be written to the
                                first edf and channels [2,4] will be written
                                to the second EDF. Any valid list of the edf
                                header can be used to split with (e.g.
                                names, transducers. etc..)
            writepaths (list):  list of Path instances specifying write
                                location of each split EDF. If None,
                                writepaths will be built from the readpath
                                by appending the file index to the readpath
                                (e.g. <readpath>_0.edf). If provided the
                                number of writepaths must match the number
                                of sublist in groups. (Default None)
            kwargs:             passed to Writer's write method.
        """

        if writepaths: 
            if len(writepaths) != len(groups):
                msg = ('Number of paths to write {} does not match ',
                        'number of EDFs to be written {}')
                raise ValueError(msg)
        else:
            writepaths = [None] * len(groups)
        for idx, (values, path) in enumerate(zip(groups, writepaths)):
            path = self._createpath(idx) if not path else path
            print('Writing File {} of {}'.format(idx+1, len(groups)))
            self._split(by, values, path, **kwargs)


if __name__ == '__main__':


    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    path2 = '/home/matt/python/nri/data/openseize/test_write.edf'
    
    """
    splitter = Splitter(path)
    splitter._split('names', ['EEG 4 SEx10', 'EEG 2 SEx10', 'EEG 3 SEx10'], None)
    """
    
    splitter = EDF(path2)
    splitter.split('channels', [[0,2], [1,3]])


