import numpy as np

from cpymad.madx import Madx

import xtrack as xt
import xobjects as xo


source = """

void get_values_at_offsets(
    CustomSetterData data,
    uint8_t* buffer,
    uint8_t* out){

    int64_t num_offsets = CustomSetterData_len_offsets(data);
    int64_t element_size = CustomSetterData_get_element_size(data);

    int64_t iout = 0;
    for (int64_t ii = 0; ii < num_offsets; ii++) {
        int64_t oo = CustomSetterData_get_offsets(data, ii);

        for (int64_t jj = 0; jj < element_size; jj++) {
            out[iout] = buffer[oo + jj];
            iout++;
        }

    }
    }

"""

class CustomSetter(xo.HybridClass):
    _xofields = {
        'element_size': xo.Int64,
        'offsets': xo.Int64[:],
    }

    _extra_c_sources = [
        source,
    ]

    _kernels = {
        'get_values_at_offsets': xo.Kernel(
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.UInt8, pointer=True, name='buffer'),
                xo.Arg(xo.UInt8, pointer=True, name='out'),
            ],
        )
    }

cs = CustomSetter(offsets=3)

prrrr

# Import SPS lattice
mad = Madx()
seq_name = 'sps'
mad.call('../../test_data/sps_w_spacecharge/sps_thin.seq')
mad.use(seq_name)
madtw = mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence[seq_name])
tracker = line.build_tracker()

# Extract list of elements to trim (all focusing quads)
elements_to_trim = [nn for nn in line.element_names if nn.startswith('qf.')]

field_to_trim = 'knl'
index_to_trim = 1

# Check all in the tracker buffer
tracker_buffer = tracker._buffer
assert np.all([line[nn]._buffer is tracker_buffer for nn in elements_to_trim])

# Extract offsets

def _extract_offset(obj, field_name, index):
    if index is None:
        return obj._xobject._get_offset(field_name)
    else:
        return getattr(obj._xobject, field_name)._get_offset(index)

offsets = [_extract_offset(line[nn], field_to_trim, index_to_trim) for nn in elements_to_trim]