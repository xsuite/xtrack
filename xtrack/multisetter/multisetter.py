import numpy as np

import xobjects as xo
import xtrack as xt

source = """

/*gpukern*/
void get_values_at_offsets(
    MultiSetterData data,
    /*gpuglmem*/ int8_t* buffer,
    /*gpuglmem*/ double* out){

    int64_t num_offsets = MultiSetterData_len_offsets(data);

    for (int64_t ii = 0; ii < num_offsets; ii++) { //vectorize_over ii num_offsets
        int64_t offs = MultiSetterData_get_offsets(data, ii);

        double val = *((/*gpuglmem*/ double*)(buffer + offs));
        out[ii] = val;
    } //end_vectorize
}

/*gpukern*/
void set_values_at_offsets(
    MultiSetterData data,
    /*gpuglmem*/ int8_t* buffer,
    /*gpuglmem*/ double* input){

    int64_t num_offsets = MultiSetterData_len_offsets(data);

    for (int64_t ii = 0; ii < num_offsets; ii++) {  //vectorize_over ii num_offsets
        int64_t offs = MultiSetterData_get_offsets(data, ii);

        double val = input[ii];
        *((/*gpuglmem*/ double*)(buffer + offs)) = val;
    } //end_vectorize
}

"""

class MultiSetter(xo.HybridClass):
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

    def __init__(self, line, elements, field, index=None):

        '''
        Create object to efficiently set and get values of a specific field of
        several elements of a line.

        Parameters
        ----------
        line: xtrack.Line
            Line in which the elements are mutated
        elements: list of int or strings
            List of indeces or names of the elements to be mutated.
        field: str
            Name of the field to be mutated.
        index: int or None
            If the field is an array, the index of the array to be mutated.

        '''

        if isinstance(line, xt.Tracker):
            tracker = line
        else:
            tracker = line.tracker

        context = tracker._context

        tracker_buffer = tracker._buffer
        line = tracker.line
        assert np.all([line[nn]._buffer is tracker_buffer for nn in elements])
        offsets = [_extract_offset(line[nn], field, index) for nn in elements]

        self.xoinitialize(_context=context, offsets=offsets)
        self.compile_kernels(only_if_needed=True)

        self.offsets = context.nparray_to_context_array(np.array(offsets))
        self.tracker = tracker
        self._tracker_buffer = tracker_buffer

    def get_values(self):

        '''
        Get the values of the multisetter fields.
        '''

        out = self._context.zeros(len(self.offsets), dtype=np.float64)
        kernel = self._context.kernels.get_values_at_offsets
        kernel.set_n_threads(len(self.offsets))
        kernel(data=self, buffer=self._tracker_buffer.buffer, out=out)
        return out

    def set_values(self, values):

        '''
        Set the values of the multisetter fields.

        Parameters
        ----------
        values: np.ndarray
            Array of values to be set.
        '''

        kernel = self._context.kernels.set_values_at_offsets
        kernel.set_n_threads(len(self.offsets))
        kernel(data=self, buffer=self._tracker_buffer.buffer,
               input=xt.BeamElement._arr2ctx(self,values))


def _extract_offset(obj, field_name, index):
    if index is None:
        assert isinstance(getattr(obj, field_name), np.float64), (
            'Only float64 fields are supported for now')
        return obj._xobject._get_offset(field_name)
    else:
        assert getattr(obj._xobject, field_name)._itemtype is xo.Float64, (
            'Only Float64 arrays are supported for now')
        return getattr(obj._xobject, field_name)._get_offset(index)