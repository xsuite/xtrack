import numpy as np

from cpymad.madx import Madx

import xtrack as xt
import xobjects as xo


# TODO: vectorize on GPU

source = """

void get_values_at_offsets(
    CustomSetterData data,
    int8_t* buffer,
    double* out){

    int64_t num_offsets = CustomSetterData_len_offsets(data);

    int64_t iout = 0;
    for (int64_t ii = 0; ii < num_offsets; ii++) {
        int64_t offs = CustomSetterData_get_offsets(data, ii);

        double val = *((double*)(buffer + offs));
        out[iout] = val;
        iout++;
    }
}

void set_values_at_offsets(
    CustomSetterData data,
    int8_t* buffer,
    double* in){

    int64_t num_offsets = CustomSetterData_len_offsets(data);

    int64_t iin = 0;
    for (int64_t ii = 0; ii < num_offsets; ii++) {
        int64_t offs = CustomSetterData_get_offsets(data, ii);

        double val = in[iin];
        *((double*)(buffer + offs)) = val;
        iin++;
    }
}

"""

class CustomSetter(xo.HybridClass):
    _xofields = {
        'offsets': xo.Int64[:],
    }

    _extra_c_sources = [
        source,
    ]

    _kernels = {
        'get_values_at_offsets': xo.Kernel(
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Float64, pointer=True, name='out'),
            ],
        ),
        'set_values_at_offsets': xo.Kernel(
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Float64, pointer=True, name='in'),
            ],
        ),
    }

    def get_values(self, buffer):
        out = np.zeros(len(self.offsets), dtype=np.float64)
        # TODO set num_threads
        self.compile_kernels(only_if_needed=True)
        kernel = self._context.kernels.get_values_at_offsets
        kernel.description.n_threads = len(self.offsets)
        kernel(data=self, buffer=buffer, out=out)
        return out


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

cs = CustomSetter(offsets=offsets)
values = cs.get_values(tracker_buffer.buffer)

