import numpy as np

import xobjects as xo
import xtrack as xt


class MultiSetter(xo.HybridClass):
    _xofields = {
        'offsets': xo.Int64[:],
    }

    _extra_c_sources = [
        '#include "xtrack/multisetter/multisetter.h"',
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
        'get_values_at_offsets_int32': xo.Kernel(
            c_name='get_values_at_offsets_int32',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Int32, pointer=True, name='out'),
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
        'set_values_at_offsets_int32': xo.Kernel(
            c_name='set_values_at_offsets_int32',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Int32, pointer=True, name='input'),
            ],
        ),
    }

    def __init__(self, line, elements, field, index=None, dtype=np.float64,
                skip_inconsistent_type_check=False):
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
        dd = getattr(inner_obj.copy(_context=xo.context_default)._xobject, inner_name)
        if index is not None:
            dd = dd[index]
        self.dtype = type(dd) if dtype is None else dtype
        self.xodtype = {
            np.float64: xo.Float64,
            np.int64: xo.Int64,
            np.int32: xo.Int32,
        }[self.dtype]

        assert self.dtype in [np.float64, np.int64, np.int32], (
            'Only float64, int64, and int32 are supported for now')

        assert np.all([line.get(nn)._buffer is tracker_buffer for nn in elements])
        offsets = [_extract_offset(line.get(nn), field, index, self.dtype, self.xodtype,
                                   skip_inconsistent_type_check=skip_inconsistent_type_check)
                   for nn in elements]

        self.xoinitialize(_context=context, offsets=offsets)
        self.compile_kernels(only_if_needed=True)

        self.offsets = context.nparray_to_context_array(np.array(offsets))
        self.tracker = tracker
        self._tracker_buffer = tracker_buffer

        self._get_kernel = {
            np.float64: self._context.kernels.get_values_at_offsets_float64,
            np.int64: self._context.kernels.get_values_at_offsets_int64,
            np.int32: self._context.kernels.get_values_at_offsets_int32,
        }[self.dtype]

        self._set_kernel = {
            np.float64: self._context.kernels.set_values_at_offsets_float64,
            np.int64: self._context.kernels.set_values_at_offsets_int64,
            np.int32: self._context.kernels.set_values_at_offsets_int32,
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


def _extract_offset(obj, field_name, index, dtype, xodtype, skip_inconsistent_type_check=False):

    if isinstance(field_name, (list, tuple)):
        inner_obj = obj
        inner_name = field_name[-1]
        for ff in field_name[:-1]:
            inner_obj = getattr(inner_obj, ff)
    else:
        inner_obj = obj
        inner_name = field_name

    if index is None:
        inconsistent_type = not isinstance(getattr(inner_obj._xobject, inner_name), dtype)
        if skip_inconsistent_type_check:
            return -1
        else:
            assert not inconsistent_type, "Inconsistent types"
        return inner_obj._xobject._get_offset(inner_name)
    else:
        obj = getattr(inner_obj._xobject, inner_name)
        inconsistent_type = not hasattr(obj, "_itemtype") or obj._itemtype is not xodtype
        if skip_inconsistent_type_check:
            return -1
        else:
            assert not inconsistent_type, "Inconsistent types"
        return obj._get_offset(index)