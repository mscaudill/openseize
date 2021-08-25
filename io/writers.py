import numpy as np


class EDFWriter:
    """ """

    def __init__(self, path):
        """ """

        self.path = path

    def bytemap(self, dic):
        """ """

        return {'version': ([8]), 
                'patient': ([80]), 
                'recording': ([80]),
                'start_date': ([8]),
                'start_time': ([8]),
                'header_bytes': ([8]),
                'reserved_0': ([44]),
                'num_records': ([8]),
                'record_duration': ([8]),
                'num_signals': ([4]),
                'names': ([16] * dic['num_signals']),
                'transducers': ([80] * dic['num_signals']),
                'physical_dim': ([8] * dic['num_signals']),
                'physical_min': ([8] * dic['num_signals']),
                'physical_max': ([8] * dic['num_signals']),
                'digital_min': ([8] * dic['num_signals']),
                'digital_max': ([8] * dic['num_signals']),
                'prefiltering': ([80] * dic['num_signals']),
                'samples_per_record': ([8] * dic['num_signals']),
                'reserved_1': ([32] * dic['num_signals'])}

    def write_header(self, dic):
        """ """

        with open(self.path, 'wb') as fp:
            for name, bytelist in self.bytemap(dic).items():
                if len(bytelist) == 1:
                    bvalue = bytes(str(dic[name]), encoding='ascii')
                    bvalue = bvalue.ljust(bytelist[0])
                    fp.write(bvalue)
                else:
                    for idx, n in enumerate(bytelist):
                        bvalue = bytes(str(dic[name][idx]), 
                                       encoding='ascii')
                        bvalue = bvalue.ljust(n)
                        fp.write(bvalue)


if __name__ == '__main__':

    from openseize.io import headers

    path = '/home/matt/python/nri/data/openseize/CW0259_P039.edf'
    header = headers.EDFHeader(path)
    
    writer = EDFWriter('sandbox/test_write2.edf')
    writer.write_header(header)
