"""A collection of functions that return dictionaries containing the
locations and datatypes in the header section of an eeg file.

The header of a file is partitioned into sections. Each section
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
apply to the read bytes.

This module contains functions that generate a bytemap for each specific
eeg file type supported by openseize. Current formats include: EDF.
"""

def edf(num_signals):
    """Returns an EDF file specific bytemap.

    Args:
        num_signals: int
        The number of signals, including possible annotations if EDF+
        format, contained in the file.

    The number of bytes of some of the sections in the header depend on the
    number of signals the EDF data records section contains.

    The EDF file specification defining this bytemap can be found @
    https://www.edfplus.info/specs/edf.html

    Returns:
        A dictionary keyed on EDF header field names with tuple
        values specifying the number of bytes to read from the last byte
        position and the type casting that should be applied to the read 
        bytes.
    """

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

