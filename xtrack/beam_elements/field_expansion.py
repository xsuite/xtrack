# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import xobjects as xo

from ..base_element import BeamElement


_FIELD_VALUE_NAMES = (
    'phi',
    'Bx', 'By', 'Bs',
    'Ax', 'Ay', 'As',
    'dAx_dx', 'dAx_dy', 'dAx_ds',
    'dAs_dx', 'dAs_dy', 'dAs_ds',
)
_FIELD_VALUE_DTYPE = np.dtype([
    (name, np.float64) for name in _FIELD_VALUE_NAMES
])


def _field_expansion_get_field(element, x, y, s):
    """Run a field-expansion evaluation kernel on broadcast input arrays."""
    context = element._context

    def _to_numpy(value):
        if isinstance(value, context.nplike_array_type):
            value = context.nparray_from_context_array(value)
        return np.asarray(value, dtype=np.float64)

    x_arr, y_arr, s_arr = np.broadcast_arrays(
        _to_numpy(x), _to_numpy(y), _to_numpy(s))
    output_shape = x_arr.shape
    n_points = x_arr.size

    if not element.straight and np.any(1.0 + element.h * x_arr == 0.0):
        raise ValueError("x contains a point on the singular curved coordinate axis")

    field_values = context.zeros(
        n_points * len(_FIELD_VALUE_NAMES), dtype=np.float64)

    if n_points:
        x_context = context.nparray_to_context_array(
            np.ascontiguousarray(x_arr).reshape(-1))
        y_context = context.nparray_to_context_array(
            np.ascontiguousarray(y_arr).reshape(-1))
        s_context = context.nparray_to_context_array(
            np.ascontiguousarray(s_arr).reshape(-1))

        n_values = int(element._ncoef) * int(element._nm)
        work_v = context.zeros(n_points * n_values, dtype=np.float64)
        work_d1 = context.zeros(n_points * n_values, dtype=np.float64)
        work_d2 = context.zeros(n_points * n_values, dtype=np.float64)
        work_q = context.zeros(n_points * int(element._nq), dtype=np.float64)

        element.compile_kernels(only_if_needed=True)
        kernel = context.kernels[element._field_evaluation_kernel_name]
        kernel(
            el=element,
            x=x_context,
            y=y_context,
            s=s_context,
            n_points=n_points,
            field_values=field_values,
            work_v=work_v,
            work_d1=work_d1,
            work_d2=work_d2,
            work_q=work_q,
        )

    field_values = context.nparray_from_context_array(field_values)
    return np.asarray(field_values).view(_FIELD_VALUE_DTYPE).reshape(output_shape)


class StraightFieldExpansion(BeamElement):
    """
    Specifies the field expansion in general derivatives on axis in straight frame.

    Parameters
        ----------
        a : array, shape na, deg+1, floats
            describing the polynomial coefficients for the skew multipoles. First index is multipole order, second index is polynomial coefficient.
        b : array, shape nb, deg+1, floats
            describing the polynomial coefficients for the normal multipoles. First index is multipole order, second index is polynomial coefficient.
        bs : array, shape deg+1, floats
            describing the polynomial coefficients for the longitudinal field component. Index is polynomial coefficient,
        ny : int
            number of powers in y to include,

    """

    isthick = True
    behaves_like_drift = True
    has_backtrack = True
    allow_loss_refinement = False
    allow_rot_and_shift = False

    _xofields = {
        "length" : xo.Float64,
        "h": xo.Float64,
        "straight": xo.Int64,
        "ny": xo.Int64,
        "deg": xo.Int64,
        "nstep": xo.Int64,
        "ds": xo.Float64,

        "a": xo.Float64[:],
        "b": xo.Float64[:],
        "bs": xo.Float64[:],

        "na": xo.Int64,
        "nb": xo.Int64,
        "deg": xo.Int64,

        "_ncoef": xo.Int64,
        "_mmax": xo.Int64,
        "_mmin": xo.Int64,
        "_moff": xo.Int64,
        "_nm": xo.Int64,

        "_qemin": xo.Int64,
        "_nq": xo.Int64,

        "_c": xo.Float64[:],
        "_V": xo.Float64[:],
        "_D1": xo.Float64[:],
        "_D2": xo.Float64[:],
        "_Q": xo.Float64[:],

        "pkin_const": xo.Int64,
        "sstart": xo.Float64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/track_fieldexpansion_straight.h"',
        '#include "xtrack/beam_elements/elements_src/create_fieldexpansion_straight.h"',
    ]

    _field_evaluation_kernel_name = 'StraightFieldExpansion_get_field'

    _kernels = {'build_expansion_straight': xo.Kernel(
            c_name='build_expansion_straight',
            args=[xo.Arg(xo.ThisClass, name='el')]
        ),
        'StraightFieldExpansion_get_field': xo.Kernel(
            c_name='StraightFieldExpansion_get_field',
            args=[
                xo.Arg(xo.ThisClass, name='el'),
                xo.Arg(xo.Float64, pointer=True, const=True, name='x'),
                xo.Arg(xo.Float64, pointer=True, const=True, name='y'),
                xo.Arg(xo.Float64, pointer=True, const=True, name='s'),
                xo.Arg(xo.Int64, name='n_points'),
                xo.Arg(xo.Float64, pointer=True, name='field_values'),
                xo.Arg(xo.Float64, pointer=True, name='work_v'),
                xo.Arg(xo.Float64, pointer=True, name='work_d1'),
                xo.Arg(xo.Float64, pointer=True, name='work_d2'),
                xo.Arg(xo.Float64, pointer=True, name='work_q'),
            ],
            n_threads='n_points',
        ),
    }

    def __init__(self, length, a, b, bs, ny, nstep=10, sstart=0, **kwargs):
        kwargs['length'] = length
        kwargs['h'] = 0
        kwargs['straight'] = 1
        kwargs['nstep'] = nstep
        kwargs['ds'] = length/nstep
        kwargs['sstart'] = sstart

        kwargs['a'] = np.asarray(a, dtype=np.float64).flatten()
        kwargs['b'] = np.asarray(b, dtype=np.float64).flatten()
        kwargs['bs'] = np.asarray(bs, dtype=np.float64)

        kwargs['na'] = a.shape[0]
        kwargs['nb'] = b.shape[0]
        kwargs['ny'] = ny

        kwargs['deg'] = a.shape[1] - 1

        if "pkin_const" in kwargs:
            kwargs["pkin_const"] = int(kwargs["pkin_const"])
        else:
            kwargs["pkin_const"] = 0  # Default is symplectic option

        if b.shape[1] != kwargs['deg'] + 1 or bs.shape[0] != kwargs['deg'] + 1:
            raise ValueError("Invalid input shapes")

        kwargs['_ncoef'] = kwargs['ny'] + 2  # store phi_0..phi_{ny+1} so By is also order ny

        kwargs['_mmax'] = kwargs['na'] if kwargs['na'] > (kwargs['nb'] - 1) else (kwargs['nb'] - 1)
        if kwargs['straight']:
            kwargs['_mmin'] = 0
            kwargs['_moff'] = 0
            kwargs['_qemin'] = 0
        else:
            kwargs['_mmin'] = -2 * ((kwargs['_ncoef'] - 1) // 2)
            kwargs['_moff'] = -kwargs['_mmin']
            kwargs['_qemin'] = kwargs['_mmin'] - 1
        kwargs['_nq'] = (kwargs['_mmax'] + 2) - kwargs['_qemin'] + 1
        kwargs['_nm'] = kwargs['_mmax'] - kwargs['_mmin'] + 1

        kwargs.setdefault("_c", np.zeros(kwargs['_ncoef'] * kwargs['_nm'] * (kwargs['deg'] + 1)))

        kwargs.setdefault("_V", np.zeros(kwargs['_ncoef'] * kwargs['_nm']))
        kwargs.setdefault("_D1", np.zeros(kwargs['_ncoef'] * kwargs['_nm']))
        kwargs.setdefault("_D2", np.zeros(kwargs['_ncoef'] * kwargs['_nm']))
        kwargs.setdefault("_Q", np.zeros(kwargs['_nq']))

        super().__init__(**kwargs)

        self.build_expansion_straight(el=self)

    def get_field(self, x, y, s):
        """Evaluate ``FieldValue`` at broadcastable ``x, y, s``.

        Returns a structured NumPy array with the broadcast input shape and
        fields ``phi``, ``Bx``, ``By``, ``Bs``, ``Ax``, ``Ay``, ``As``, and
        all derivatives stored by the C ``FieldValue`` structure.
        """
        return _field_expansion_get_field(self, x=x, y=y, s=s)

class BentFieldExpansion(BeamElement):
    """
    Specifies the field expansion in general derivatives on axis in curved frame.

    Parameters
        ----------
        h : float
            Curvature of the element, in 1/m. For straight elements, h=0, use StraightFieldExpansion.
        a : array, shape na, deg+1, floats
            describing the polynomial coefficients for the skew multipoles. First index is multipole order, second index is polynomial coefficient.
        b : array, shape nb, deg+1, floats
            describing the polynomial coefficients for the normal multipoles. First index is multipole order, second index is polynomial coefficient.
        bs : array, shape deg+1, floats
            describing the polynomial coefficients for the longitudinal field component. Index is polynomial coefficient,
        ny : int
            number of powers in y to include,

    """

    isthick = True
    behaves_like_drift = True
    has_backtrack = True
    allow_loss_refinement = False
    allow_rot_and_shift = False

    _xofields = {
        "length" : xo.Float64,
        "h": xo.Float64,
        "straight": xo.Int64,
        "ny": xo.Int64,
        "deg": xo.Int64,
        "nstep": xo.Int64,
        "ds": xo.Float64,

        "a": xo.Float64[:],
        "b": xo.Float64[:],
        "bs": xo.Float64[:],

        "na": xo.Int64,
        "nb": xo.Int64,
        "deg": xo.Int64,

        "_ncoef": xo.Int64,
        "_mmax": xo.Int64,
        "_mmin": xo.Int64,
        "_moff": xo.Int64,
        "_nm": xo.Int64,

        "_qemin": xo.Int64,
        "_nq": xo.Int64,

        "_c": xo.Float64[:],
        "_V": xo.Float64[:],
        "_D1": xo.Float64[:],
        "_D2": xo.Float64[:],
        "_Q": xo.Float64[:],

        "pkin_const": xo.Int64,
        "sstart": xo.Float64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/track_fieldexpansion_bent.h"',
        '#include "xtrack/beam_elements/elements_src/create_fieldexpansion_bent.h"',
    ]

    _field_evaluation_kernel_name = 'BentFieldExpansion_get_field'

    _kernels = {'build_expansion_bent': xo.Kernel(
            c_name='build_expansion_bent',
            args=[xo.Arg(xo.ThisClass, name='el')]
        ),
        'BentFieldExpansion_get_field': xo.Kernel(
            c_name='BentFieldExpansion_get_field',
            args=[
                xo.Arg(xo.ThisClass, name='el'),
                xo.Arg(xo.Float64, pointer=True, const=True, name='x'),
                xo.Arg(xo.Float64, pointer=True, const=True, name='y'),
                xo.Arg(xo.Float64, pointer=True, const=True, name='s'),
                xo.Arg(xo.Int64, name='n_points'),
                xo.Arg(xo.Float64, pointer=True, name='field_values'),
                xo.Arg(xo.Float64, pointer=True, name='work_v'),
                xo.Arg(xo.Float64, pointer=True, name='work_d1'),
                xo.Arg(xo.Float64, pointer=True, name='work_d2'),
                xo.Arg(xo.Float64, pointer=True, name='work_q'),
            ],
            n_threads='n_points',
        ),
    }

    def __init__(self, length, h, a, b, bs, ny, nstep=10, sstart=0, **kwargs):
        assert h > 1e-4, "Use straight element with h=0!"
        kwargs['length'] = length
        kwargs['h'] = h
        kwargs['straight'] = 0
        kwargs['nstep'] = nstep
        kwargs['ds'] = length/nstep
        kwargs['sstart'] = sstart

        kwargs['a'] = np.asarray(a, dtype=np.float64).flatten()
        kwargs['b'] = np.asarray(b, dtype=np.float64).flatten()
        kwargs['bs'] = np.asarray(bs, dtype=np.float64)

        kwargs['na'] = a.shape[0]
        kwargs['nb'] = b.shape[0]
        kwargs['ny'] = ny

        kwargs['deg'] = a.shape[1] - 1

        if "pkin_const" in kwargs:
            kwargs["pkin_const"] = int(kwargs["pkin_const"])
        else:
            kwargs["pkin_const"] = 0  # Default symplectic option


        if b.shape[1] != kwargs['deg'] + 1 or bs.shape[0] != kwargs['deg'] + 1:
            raise ValueError("Invalid input shapes")

        kwargs['_ncoef'] = kwargs['ny'] + 2  # store phi_0..phi_{ny+1} so By is also order ny

        kwargs['_mmax'] = kwargs['na'] if kwargs['na'] > (kwargs['nb'] - 1) else (kwargs['nb'] - 1)
        if kwargs['straight']:
            kwargs['_mmin'] = 0
            kwargs['_moff'] = 0
            kwargs['_qemin'] = 0
        else:
            kwargs['_mmin'] = -2 * ((kwargs['_ncoef'] - 1) // 2)
            kwargs['_moff'] = -kwargs['_mmin']
            kwargs['_qemin'] = kwargs['_mmin'] - 1
        kwargs['_nq'] = (kwargs['_mmax'] + 2) - kwargs['_qemin'] + 1
        kwargs['_nm'] = kwargs['_mmax'] - kwargs['_mmin'] + 1

        kwargs.setdefault("_c", np.zeros(kwargs['_ncoef'] * kwargs['_nm'] * (kwargs['deg'] + 1)))

        kwargs.setdefault("_V", np.zeros(kwargs['_ncoef'] * kwargs['_nm']))
        kwargs.setdefault("_D1", np.zeros(kwargs['_ncoef'] * kwargs['_nm']))
        kwargs.setdefault("_D2", np.zeros(kwargs['_ncoef'] * kwargs['_nm']))
        kwargs.setdefault("_Q", np.zeros(kwargs['_nq']))

        super().__init__(**kwargs)

        self.build_expansion_bent(el=self)

    def get_field(self, x, y, s):
        """Evaluate ``FieldValue`` at broadcastable ``x, y, s``.

        Returns a structured NumPy array with the broadcast input shape and
        fields ``phi``, ``Bx``, ``By``, ``Bs``, ``Ax``, ``Ay``, ``As``, and
        all derivatives stored by the C ``FieldValue`` structure.
        """
        return _field_expansion_get_field(self, x=x, y=y, s=s)


class FieldExpansion(BeamElement):
    def __new__(cls, *args, **kwargs):
        if 'h' in kwargs and kwargs['h'] > 1e-9:
            return BentFieldExpansion(*args, **kwargs)
        else:
            kwargs.pop('h', None)
            return StraightFieldExpansion(*args, **kwargs)
