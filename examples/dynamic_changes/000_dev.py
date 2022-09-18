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
    for (int64_t ii = 0; ii < num_offsets; ii++) { //vectorize_over ii num_offsets
        int64_t offs = CustomSetterData_get_offsets(data, ii);

        double val = *((double*)(buffer + offs));
        out[iout] = val;
        iout++;
    } //end_vectorize
}

void set_values_at_offsets(
    CustomSetterData data,
    int8_t* buffer,
    double* input){

    int64_t num_offsets = CustomSetterData_len_offsets(data);

    int64_t iin = 0;
    for (int64_t ii = 0; ii < num_offsets; ii++) {  //vectorize_over ii num_offsets
        int64_t offs = CustomSetterData_get_offsets(data, ii);

        double val = input[iin];
        *((double*)(buffer + offs)) = val;
        iin++;
    } //end_vectorize
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
                xo.Arg(xo.Float64, pointer=True, name='input'),
            ],
        ),
    }

    def __init__(self, tracker, elements, field, index=None):

        context = tracker._context

        tracker_buffer = tracker._buffer
        line = tracker.line
        assert np.all([line[nn]._buffer is tracker_buffer for nn in elements])
        offsets = [_extract_offset(line[nn], field, index) for nn in elements]

        self.xoinitialize(_context=context, offsets=offsets)

        self.offsets = offsets
        self.tracker = tracker
        self._tracker_buffer = tracker_buffer

    def get_values(self):
        out = np.zeros(len(self.offsets), dtype=np.float64)
        self.compile_kernels(only_if_needed=True)
        kernel = self._context.kernels.get_values_at_offsets
        kernel.description.n_threads = len(self.offsets)
        kernel(data=self, buffer=self._tracker_buffer.buffer, out=out)
        return out

    def set_values(self, values):
        self.compile_kernels(only_if_needed=True)
        kernel = self._context.kernels.set_values_at_offsets
        kernel.description.n_threads = len(self.offsets)
        kernel(data=self, buffer=self._tracker_buffer.buffer, input=values)

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

def _extract_offset(obj, field_name, index):
    if index is None:
        return obj._xobject._get_offset(field_name)
    else:
        return getattr(obj._xobject, field_name)._get_offset(index)


cs = CustomSetter(tracker=tracker, elements=elements_to_trim,
                  field=field_to_trim, index=index_to_trim)
values = cs.get_values()
cs.set_values(values*1.1)

