import numpy as np

import xobjects as xo
import xtrack as xt

source = """

/*gpukern*/
void get_values_at_offsets_float64(
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
void get_values_at_offsets_int64(
    MultiSetterData data,
    /*gpuglmem*/ int8_t* buffer,
    /*gpuglmem*/ int64_t* out){

    int64_t num_offsets = MultiSetterData_len_offsets(data);

    for (int64_t ii = 0; ii < num_offsets; ii++) { //vectorize_over ii num_offsets
        int64_t offs = MultiSetterData_get_offsets(data, ii);

        int64_t val = *((/*gpuglmem*/ int64_t*)(buffer + offs));
        out[ii] = val;
    } //end_vectorize
}

/*gpukern*/
void set_values_at_offsets_float64(
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

/*gpukern*/
void set_values_at_offsets_int64(
    MultiSetterData data,
    /*gpuglmem*/ int8_t* buffer,
    /*gpuglmem*/ int64_t* input){

    int64_t num_offsets = MultiSetterData_len_offsets(data);

    for (int64_t ii = 0; ii < num_offsets; ii++) {  //vectorize_over ii num_offsets
        int64_t offs = MultiSetterData_get_offsets(data, ii);

        int64_t val = input[ii];
        *((/*gpuglmem*/ int64_t*)(buffer + offs)) = val;
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
        'get_values_at_offsets_float64': xo.Kernel(
            c_name='get_values_at_offsets_float64',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Float64, pointer=True, name='out'),
            ],
        ),
        'get_values_at_offsets_int64': xo.Kernel(
            c_name='get_values_at_offsets_int64',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Int64, pointer=True, name='out'),
            ],
        ),
        'set_values_at_offsets_float64': xo.Kernel(
            c_name='set_values_at_offsets_float64',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Float64, pointer=True, name='input'),
            ],
        ),
        'set_values_at_offsets_int64': xo.Kernel(
            c_name='set_values_at_offsets_int64',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Int64, pointer=True, name='input'),
            ],
        ),
    }

    def __init__(self, line, elements, field, index=None):
        """Create object to efficiently set and get values of a specific field of
        several elements of a line.

        Parameters
        ----------
        line: xtrack.Line
            Line in which the elements are mutated
        elements: list of int or strings
            List of indices or names of the elements to be mutated.
        field: str
            Name of the field to be mutated.
        index: int or None
            If the field is an array, the index of the array to be mutated.
        """

        if isinstance(line, xt.Tracker):
            tracker = line
        else:
            tracker = line.tracker

        if tracker.iscollective:
            tracker = tracker.line._get_non_collective_line().tracker

        context = tracker._context

        tracker_buffer = tracker._buffer
        line = tracker.line

        if len(elements) == 0:
            self._empty = True
            self.xoinitialize(_context=context, offsets=[])
            return

        self._empty = False

        # Get dtype from first element
        el = line[elements[0]]
        if isinstance(field, (list, tuple)):
            inner_obj = el
            inner_name = field[-1]
            for ff in field[:-1]:
                inner_obj = getattr(inner_obj, ff)
        else:
            inner_obj = el
            inner_name = field
        dd = getattr(inner_obj.copy(_context=xo.context_default), inner_name)
        if index is not None:
            dd = dd[index]
        self.dtype = type(dd)
        self.xodtype = {
            np.float64: xo.Float64,
            np.int64: xo.Int64,
        }[self.dtype]

        assert self.dtype in [np.float64, np.int64], (
            'Only float64 and int64 are supported for now')

        assert np.all([line[nn]._buffer is tracker_buffer for nn in elements])
        offsets = [_extract_offset(line[nn], field, index, self.dtype, self.xodtype)
                   for nn in elements]

        self.xoinitialize(_context=context, offsets=offsets)
        self.compile_kernels(only_if_needed=True)

        self.offsets = context.nparray_to_context_array(np.array(offsets))
        self.tracker = tracker
        self._tracker_buffer = tracker_buffer

        self._get_kernel = {
            np.float64: self._context.kernels.get_values_at_offsets_float64,
            np.int64: self._context.kernels.get_values_at_offsets_int64,
        }[self.dtype]

        self._set_kernel = {
            np.float64: self._context.kernels.set_values_at_offsets_float64,
            np.int64: self._context.kernels.set_values_at_offsets_int64,
        }[self.dtype]

    def get_values(self):
        """Get the values of the multisetter fields."""
        if self._empty:
            return self._context.zeros(0, dtype=np.float64)

        out = self._context.zeros(len(self.offsets), dtype=self.dtype)
        self._get_kernel.set_n_threads(len(self.offsets))
        self._get_kernel(data=self, buffer=self._tracker_buffer.buffer, out=out)
        return out

    def set_values(self, values):
        """Set the values of the multisetter fields.

        Parameters
        ----------
        values: np.ndarray
            Array of values to be set.
        """
        if self._empty:
            return

        self._set_kernel.set_n_threads(len(self.offsets))
        self._set_kernel(data=self, buffer=self._tracker_buffer.buffer,
               input=xt.BeamElement._arr2ctx(self, values))

    def compile_kernels(self, only_if_needed=True):
        context = self._buffer.context
        if context.allow_prebuilt_kernels and only_if_needed:
            try:
                from xsuite import (
                    get_suitable_kernel,
                    XSK_PREBUILT_KERNELS_LOCATION,
                )
                kernel_info = get_suitable_kernel({}, ())
            except ImportError:
                kernel_info = None

            if kernel_info:
                module_name, _ = kernel_info
                kernels = context.kernels_from_file(
                    module_name=module_name,
                    containing_dir=XSK_PREBUILT_KERNELS_LOCATION,
                    kernel_descriptions=self._kernels,
                )
                context.kernels.update(kernels)

        super().compile_kernels(only_if_needed=only_if_needed)


def _extract_offset(obj, field_name, index, dtype, xodtype):

    if isinstance(field_name, (list, tuple)):
        inner_obj = obj
        inner_name = field_name[-1]
        for ff in field_name[:-1]:
            inner_obj = getattr(inner_obj, ff)
    else:
        inner_obj = obj
        inner_name = field_name

    if index is None:
        assert isinstance(getattr(inner_obj, inner_name), dtype), (
            "Inconsistent types")
        return inner_obj._xobject._get_offset(inner_name)
    else:
        assert getattr(inner_obj._xobject, inner_name)._itemtype is xodtype, (
            "Inconsistent types")
        return getattr(inner_obj._xobject, inner_name)._get_offset(index)