import numpy as np

from openseize.io import bases


class Header(bases.Header):
    """An extended dictionary representation of an EDF Header."""

    def bytemap(self, num_signals=None):
        """Specifies the number of bytes to sequentially read for each field
        in an EDF header and dataype conversions to apply.

        The header of an EDF file is partitioned into sections. Each section
        spans a number of bytes and contains a specific piece of header
        data. Below is the first two sections of an .edf file header:

        *************************************** 
        * 8 bytes ** 80 bytes .................
        ***************************************

        The first 8 bytes correspond to the edf version string and the next
        80 bytes corresponds to a patient id string. A bytemap specifies the
        name, number of bytes, and datatype as dict of tuples like so:

        {'version': (8, str), 'patient': (80, str), ....}

        This mapping defines the name of what is read, the number of bytes
        to read (relative to last byte position) and the type casting to
        apply to the read bytes. The number of bytes of some of the sections 
        in the header depend on the number of signals the EDF data records.

        The EDF file specification defining this bytemap can be found @
        https://www.edfplus.info/specs/edf.html

        Args:
            num_signals: int
                The number of signals (channels & annotation) in the file.
                If None the number of signals will be read automatically
                from the opened path instance. Default is to read this
                automatically.

        Returns: 
            A dictionary keyed on EDF specification field names with tuple
            values specifying the number of bytes to read from the last byte
            position and the type casting that should be applied to the read 
            bytes.
        """

        if num_signals is None:
            num_signals = self.count_signals()
        
        return {'version': ([8], str), 
                'patient': ([80], str), 
                'recording': ([80], str),
                'start_date': ([8], str),
                'start_time': ([8], str),
                'header_bytes': ([8], int),
                'reserved_0': ([44], str),
                'num_records': ([8], int),
                'record_duration': ([8], float),
                'num_signals': ([4], int),
                'names': ([16] * num_signals, str),
                'transducers': ([80] * num_signals, str),
                'physical_dim': ([8] * num_signals, str),
                'physical_min': ([8] * num_signals, float),
                'physical_max': ([8] * num_signals, float),
                'digital_min': ([8] * num_signals, float),
                'digital_max': ([8] * num_signals, float),
                'prefiltering': ([80] * num_signals, str),
                'samples_per_record': ([8] * num_signals, int),
                'reserved_1': ([32] * num_signals, str)}

    def count_signals(self):
        """Returns the number of signals, including possible annotations,
        in this EDF."""

        with open(self.path, 'rb') as fp:
            fp.seek(252) # edf specifies num signals at 252nd byte
            return int(fp.read(4).strip().decode())

    @classmethod
    def from_dict(cls, dic):
        """Alternative constructor that creates a Header instance from
        a dictionary.

        Args:
            dic: dictionary 
                A dictionary containing all expected bytemap keys.
        """

        instance = cls(path=None)
        instance.update(dic)
        # validate dic contains all bytemap keys
        if set(dic) == set(self.bytemap(1)):
            return instance
        else:
            msg='Missing keys required to create a header of type {}.'
            raise ValueError(msg.format(cls.__name__))

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
    def samples(self):
        """Returns summed sample count across records for each channel.

        The last record of the EDF may not be completely filled with
        recorded signal values depending on the software that created it.
        """

        nrecs = self.num_records
        samples = np.array(self.samples_per_record) * nrecs
        return [samples[ch] for ch in self.channels]

    @property
    def record_map(self):
        """Returns a list of slices corresponding to the start, stop sample
        indices within a record for each channel.

        Within a record each channel is sequentially listed like so:

        Record:
        ******************************************************
        * Ch0 samples, Ch1 samples, Ch2 samples, Ch3 samples *
        ******************************************************

        This function returns the start, stop indices for each channel as
        a list of slices.
        """

        scnts = np.insert(self.samples_per_record,0,0)
        cum = np.cumsum(scnts)
        return list(slice(a, b) for (a, b) in zip(cum, cum[1:]))

    @property
    def slopes(self):
        """Returns the slope (gain) of each channel in this Header.

        The EDF file specification asserts that the physical voltage 
        values 'p' are linearly mapped from the integer digital values 'd'
        in the EDF according to:

            p = slope * d + offset
            slope = (pmax -pmin) / (dmax - dmin)
            offset = p - slope * d for any (p,d)
        
        This function computes the slope term from the physical and digital
        values stored in the header for each channel.

        Returns: 
            A 1-D array of slope values one per signal in EDF.
        """

        pmaxes = np.array(self.physical_max)[self.channels]
        pmins = np.array(self.physical_min)[self.channels]
        dmaxes = np.array(self.digital_max)[self.channels]
        dmins = np.array(self.digital_min)[self.channels]
        return (pmaxes - pmins) / (dmaxes - dmins)

    @property
    def offsets(self):
        """Returns the offset of each channel in this Header.

        See slopes property.

        Returns: 
            A 1-D array of offset values one per signal in EDF.
        """

        pmins = np.array(self.physical_min)[self.channels]
        dmins = np.array(self.digital_min)[self.channels]
        return (pmins - self.slopes * dmins)
    
    def filter(self, indices):
        """Filters this Header keeping only data that pertains to each
        channel index in indices.

        Args:
            indices: sequence of ints
                Integer channel indices to retain in this Header.

        Returns: 
            A new header instance.
        """
        
        header = copy.deepcopy(self)
        for key, value in header.items():
            if isinstance(value, list):
                header[key] = [value[idx] for idx in indices]
        
        # update header_bytes and num_signals
        bytemap = self.bytemap(len(indices))
        nbytes = sum(sum(tup[0]) for tup in bytemap.values())
        header['header_bytes'] = nbytes
        header['num_signals'] = len(indices)

        return header


class Reader(bases.Reader):
    """A reader of European Data Format (EDF/EDF+) files.

    The EDF specification has a header section followed by data records
    Each data record contains all signals stored sequentially. EDF+
    files include an annotation signal within each data record. To
    distinguish these signals we refer to data containing signals as
    channels and annotation signals as annotation. Currently, this reader
    does not support the reading of annotation signals.

    For details on the EDF/+ file specification please see:

    https://www.edfplus.info/specs/index.html

    Attributes:
        header: A dictionary representation of an EDF Header.
        shape: A tuple of channels, samples contained in this EDF
    """

    def __init__(self, path):
        """Extends the Reader ABC with a header attribute."""

        super().__init__(path, mode='rb')
        self.header = Header(path)

    @property
    def shape(self):
        """Returns a 2-tuple containing the number of channels and 
        number of samples in this EDF."""

        return len(self.header.channels), max(self.header.samples)

    def _decipher(self, arr, channels, axis=-1):
        """Converts an array of EDF integers to an array of voltage floats.

        The EDF file specification asserts that the physical voltage 
        values 'p' are linearly mapped from the integer digital values 'd'
        in the EDF according to:

            p = slope * d + offset
            slope = (pmax -pmin) / (dmax - dmin)
            offset = p - slope * d for any (p,d)

        The EDF header contains pmax, pmin, dmax and dmin for each channel.

        Args:
            arr: 2-D array
                An array of integer digital values read from the EDF file.
            channels: sequence
                Sequence of channels read from the EDF file
            axis: int
                The samples axis of arr. Default is last axis.
            
        Returns: 
            A float64 ndarray of physical voltage values with shape matching
            input 'arr' shape.
        """

        slopes = self.header.slopes[channels]
        offsets = self.header.offsets[channels]
        #expand to 2-D for broadcasting
        slopes = np.expand_dims(slopes, axis=axis)
        offsets = np.expand_dims(offsets, axis=axis)
        result = arr * slopes
        result += offsets
        return result

    def _find_records(self, start, stop, channels):
        """Locates a start and stop record number containing the start and
        stop sample number for each channel in channels.

        EDF files are partitioned into records. Each record contains data
        for each channel sequentially. Below is an example record for
        4-channels of data.

        Record:
        ******************************************************
        * Ch0 samples, Ch1 samples, Ch2 samples, Ch3 samples *
        ******************************************************

        The number of samples for each chanel will be different if the
        sample rates for the channels are not equal. The number of samples
        in a record for each channel is given in the header by the field
        samples_per_record. This method locates the start and stop record
        numbers that include the start and stop sample indices for each
        channel.

        Args:
            start: int
                The start sample to read.
            stop: int
                The stop sample to read (exclusive).
            channels: sequence
                Sequence of channels to read.

        Returns: 
            A list of 2-tuples containing the start and stop record numbers
            that inlcude the start and stop sample number for each channel
            in channels.
        """

        spr = np.array(self.header.samples_per_record)[channels]
        starts = start // spr
        stops = np.ceil(stop / spr).astype('int')
        return list(zip(starts, stops))

    def _records(self, a, b):
        """Reads samples between the ath to bth record.

        If b exceeds the number of records in the EDF, then samples upto the
        end of file are returned. If a exceeds the number of records, an
        empty array is returned.

        Args:
            a: int
                The start record to read.
            b: int
                The last record to be read (exclusive).

        Returns:
            A 2-D array of shape (b-a) x sum(samples_per_record)
        """

        if a >= self.header.num_records:
            return np.empty((1,0))
        b = min(b, self.header.num_records)
        cnt = b - a
        
        self._fobj.seek(0)
        #EDF samples are 2-byte integers
        bytes_per_record = sum(self.header.samples_per_record) * 2
        #get offset in bytes & num samples spanning a to b
        offset = self.header.header_bytes + a * bytes_per_record
        nsamples = cnt * sum(self.header.samples_per_record)
        #read records and reshape to num_records x sum(samples_per_record)
        recs = np.fromfile(self._fobj, '<i2', nsamples, offset=offset)
        arr = recs.reshape(cnt, sum(self.header.samples_per_record))
        return arr

    def _padstack(self, arrs, value, axis=0):
        """Pads a sequence of 1-D arrays to equal length and stacks them
        along axis.

        Args:
            arrs: sequence
                A sequence of 1-D arrays.
            value: float
                Value to append to arrays that are shorter than the longest
                array in arrs.
        
        Returns:
            A 2-D array.
        """

        longest = max(len(arr) for arr in arrs)
        pad_sizes = np.array([longest - len(arr) for arr in arrs])

        if all(pad_sizes == 0):
            return np.stack(arrs, axis=0)
        
        else:
            x = [np.pad(arr.astype(float), (0, pad), constant_values=value)
                    for arr, pad in zip(arrs, pad_sizes)]
            return np.stack(x, axis=0)

    def _read_array(self, start, stop, channels, padvalue):
        """Reads samples between start & stop for each channel in channels.

        Args:
            start: int
                The start sample index to read.
            stop: int
                The stop sample index to read (exclusive).
            channels: sequence
                Sequence of channels to read from EDF.
            padvalue: float
                Value to pad to channels that run out of data to return.
                Only applicable if sample rates of channels differ.

        Returns:
            A float64 2-D array of shape len(channels) x (stop-start).
        """
        
        # Locate record tuples that include start & stop samples for
        # each channel but only perform reads over unique record tuples.
        rec_tuples = self._find_records(start, stop, channels)
        uniq_tuples = set(rec_tuples)
        reads = {tup: self._records(*tup) for tup in uniq_tuples}

        result=[]
        for ch, rec_tup in zip(channels, rec_tuples):
            
            #get preread array and extract samples for this ch
            arr = reads[rec_tup]
            arr = arr[:, self.header.record_map[ch]].flatten()
            
            #adjust start & stop relative to records start pt
            a = start - rec_tup[0] * self.header.samples_per_record[ch]
            b = a + (stop - start)
            result.append(arr[a:b])
        
        res = self._padstack(result, padvalue)
        return self._decipher(res, channels)
    
    def read(self, start, stop=None, channels=None, padvalue=np.NaN):
        """Reads samples from this EDF for the specified channels.

        Args:
            start: int
                The start sample index to read.
            stop: int
                The stop sample index to read (exclusive). If None, samples
                will be read until the end of file. Default is None.
            channels: sequence
                Sequence of channels to read from EDF. If None, all channels
                in the EDF will be read. Default is None.
            padvalue: float
                Value to pad to channels that run out of samples to return.
                Only applicable if sample rates of channels differ. Default
                padvalue is NaN.

        Returns: 
            A float64 array of shape len(chs) x (stop-start) samples.
        """

        channels = self.header.channels if not channels else channels
        if start > max(self.header.samples):
            return np.empty((len(channels), 0))
        if not stop:
            stop = max(self.header.samples)
        return self._read_array(start, stop, channels, padvalue)


class Writer(bases.Writer):
    """A writer of European Data Format (EDF) files.

    This writer does not support writing annotations to an EDF file.
    """

    def __init__(self, path):
        """Initialize this Writer. See base class for futher details."""

        super().__init__(path, mode='wb')

    def _write_header(self, header):
        """Write a header dict to this Writer's opened file instance.

        Args:
            header: Header dict          
                A dict of EDF compliant metadata. Please see Header for
                further details.
        """
       
        self.header = header
        bytemap = header.bytemap(header.num_signals)
        # Move to file start and write each ascii encoded byte string
        self._fobj.seek(0)
        for items, (nbytes, _) in zip(header.values(), bytemap.values()):
            items = [items] if not isinstance(items, list) else items
            for item, nbyte in zip(items, nbytes):
                bytestr = bytes(str(item), encoding='ascii').ljust(nbyte)
                self._fobj.write(bytestr)

    def _records(self, data, channels):
        """Yields a list of sample arrays one per channel to write to
        a single data record

        Args:
            data: 2-D array, memmap, Reader instance
                An object that is sliceable or has a read method that 
                returns arrays of shape channels x samples. 
            channels: sequence
                A sequence of channels to write to each data record in this
                Writer.

        Yields:
            A list of 1-D arrays of samples for a single data record, one
            array per channel in channels.
        """

        for n in range(self.header.num_records):
            result = []
            # The number of samples per record is channel dependent if
            # sample rates are not equal across channels.
            starts = n * np.array(self.header.samples_per_record)
            stops = (n+1) * np.array(self.header.samples_per_record)

            for channel, start, stop in zip(channels, starts, stops):
                if isinstance(data, np.ndarray):
                    result.append(data[channel][start:stop])
                else:
                    result.append(data.read(start, stop, [channel]))
            
            yield result

    def _encipher(self, arrs):
        """Transforms each array in a sequence of arrays from float to
        a 2-byte little-endian integer dtype.

        Args:
            arrs: sequence
                Sequence of 1-D arrays of float dtype.

        See also _decipher method of the EDFReader.

        Returns:
            A sequence of 1-D arrays in 2-byte little-endian fmt.
        """

        slopes = self.header.slopes
        offsets = self.header.offsets
        results = []
        for ch, x in enumerate(arrs):
            arr = np.rint((x - offsets[ch]) / slopes[ch])
            arr = arr.astype('<i2')
            results.append(arr)
        return results

    def _validate(self, header, data):
        """Validates that samples in data is divisible by number of records
        in header.

        The EDF file spec. does not allow the writing of partial record at
        the end of the file. Therefore, the data to be written needs to 
        perfectly fill all num_records in the header. This may require 
        appending 0's to the end of data to ensure this.

        Args:
            header: dict
                An EDFHeader instance.
            data: 2-D array or Reader instance
                Data to be written to this Writer's open file instance.
        
        Raises:
            A ValueError if num. samples is not divisible by num. records 
            in header. 
        """

        if data.shape[1] % header.num_records != 0:
            msg=('Number of data samples must be divisible by '
                 'the number of records; {} % {} != 0')
            raise ValueError(msg.format(values, num_records))

    def _progress(self, record_idx):
        """Relays write progress during file writing."""
        
        msg = 'Writing data: {:.1f}% complete'
        perc = record_idx / self.header.num_records * 100
        print(msg.format(perc), end='\r', flush=True) 

    def write(self, header, data, channels, verbose=True):
        """Write header & data for each channel to file object.

        Args:
            header: dict
                A mapping of EDF compliant fields and values. For Further
                details see Header class of this module.
            data: 2-D array or Reader instance
                A channels x samples array or Reader instance.
            channels: sequence
                A sequence of channel indices to write to this Writer's 
                open file instance.
            verbose: bool
                An option to print progress of write. Default (True) prints
                status update as each record is written.
        """

        header = headers.EDFHeader.from_dict(header)
        header = header.filter(channels)
        self._validate(header, data)

        self._write_header(header) #and store header to instance
        self._fobj.seek(header.header_bytes)
        for idx, record in enumerate(self._records(data, channels)):
            samples = self._encipher(record) # floats to '<i2'
            samples = np.concatenate(samples)
            #concatenate data bytes to str and write
            byte_str = samples.tobytes()
            self._fobj.write(byte_str)
            if verbose:
                self._progress(idx)
