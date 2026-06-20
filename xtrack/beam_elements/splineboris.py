# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from dataclasses import dataclass

import numpy as np

import xobjects as xo

from ..base_element import BeamElement
from ..random import RandomUniformAccurate, RandomExponential
from .elements import SynchrotronRadiationRecord

_POLY_ORDER = 4
_NUM_COEFFS = _POLY_ORDER + 1
_MAX_MULTIPOLE_ORDER = 7
_HERMITE_SUFFIXES = ("val_start", "der_start", "val_end", "der_end", "mean")


@dataclass
class Spline4:
    """Hermite boundary data and interval mean used by ``SplineBoris``."""

    val_start: float
    der_start: float
    val_end: float
    der_end: float
    mean: float

    def as_list(self):
        return [
            self.val_start,
            self.der_start,
            self.val_end,
            self.der_end,
            self.mean,
        ]

    def as_dict(self):
        return {
            'val_start': self.val_start,
            'der_start': self.der_start,
            'val_end': self.val_end,
            'der_end': self.der_end,
            'mean': self.mean,
        }

    def as_np_array(self):
        return np.array(self.as_list())

# Canonical naming convention for the polynomial coefficients
# Used by _generate_bpmeth_to_C.py to generate the C code for the field evaluation.
def _get_param_names(multipole_order):
    """Ordered names of polynomial coefficients in ``par_list`` / ``evaluate_B``'s ``params``.
    This function is not used here, but is called when generating the C code for the field evaluation.
    It is kept here, so that the SplineBoris defines the canonical naming convention for the polynomial coefficients.

    Convention: ``bs_{k}``, then ``by_{i}_{k}``, then ``bx_{i}_{k}``.
    Where ``k`` is the polynomial order, ``i`` is the multipole order.
    """
    if multipole_order < 1:
        raise ValueError("multipole_order must be >= 1")
    if multipole_order > _MAX_MULTIPOLE_ORDER:
        raise ValueError(
            f"multipole_order ({multipole_order}) exceeds maximum supported "
            f"({_MAX_MULTIPOLE_ORDER})"
        )
    names = [f"bs_{k}" for k in range(_NUM_COEFFS)]
    for i in range(multipole_order):
        names.extend(f"by_{i}_{k}" for k in range(_NUM_COEFFS))
    for i in range(multipole_order):
        names.extend(f"bx_{i}_{k}" for k in range(_NUM_COEFFS))
    return names

# This function "sanitizes" the "inner" tuple of Spline4 or dicts
# For example, if bx = (Spline4, Dict, None):
# - This function is called from _sanitize_init_tuple for the "inner" tuple
# - It will see that the first element is a Spline4, and return it
# - It will see that the second element is a Dict, and convert it to a Spline4
def _sanitize_init_tuple_elements(data, name):
    if isinstance(data, Spline4):
        return data

    if isinstance(data, dict):
        required = ('val_start', 'der_start', 'val_end', 'der_end', 'mean')
        missing = [kk for kk in required if kk not in data]
        if missing:
            raise ValueError(f"{name} is missing keys: {missing}")
        return Spline4(**{kk: data[kk] for kk in required})

    if isinstance(data, (list, tuple, np.ndarray)):
        if len(data) != _NUM_COEFFS:
            raise ValueError(f"{name} must contain {_NUM_COEFFS} values, got {len(data)}")
        return Spline4(*data)

    raise TypeError(
        f"{name} must be a Spline4, dict, or list/tuple of "
        f"{_NUM_COEFFS} values; got {type(data).__name__}"
    )

# This function "sanitizes" the "outer" tuple of Spline4 or dicts
# For example, if bx = (Spline4, Dict, None):
# - Using _sanitize_init_tuple_elements, the Dict will be converted to a Spline4
# - The None will be converted to an empty tuple
# - The result will be a tuple of (Spline4, Spline4, None)
def _sanitize_init_tuple(values, name):
    """Normalize component input to a tuple of ``Spline4`` or ``None`` entries."""
    if values is None:
        return ()
    if isinstance(values, Spline4):
        values = (values,)
    elif isinstance(values, dict):
        values = (values,)
    elif isinstance(values, list):
        values = tuple(values)
    elif not isinstance(values, tuple):
        raise TypeError(
            f"{name} must be a Spline4/dict, tuple/list of Spline4/dict/None, or None; "
            f"got {type(values).__name__}"
        )
    out = []
    for order, item in enumerate(values):
        if item is None:
            out.append(None)
            continue
        out.append(_sanitize_init_tuple_elements(item, f"{name}[{order}]"))
    return tuple(out)

# This function checks if the input:
# - Finds the maximum multipole order
# - Validates that the maximum multipole order is not greater than the maximum supported (7)
# - It also checks if the number of elements in each Spline4 is _NUM_COEFFS (5)
# - It converts the tuple of Spline4 to a numpy array
# - Returns the numpy arrays and the detected multipole order
def _validate_and_convert_to_array(bs, by, bx):
    """Validate serialized inputs and build xobject-ready Hermite storage arrays."""

    by_tuple = _sanitize_init_tuple(by, "by")
    bx_tuple = _sanitize_init_tuple(bx, "bx")

    bs_spline = _sanitize_init_tuple_elements(bs, "bs")

    max_order = -1
    for order, coeffs in enumerate(by_tuple):
        if coeffs is not None:
            max_order = max(max_order, order)
    for order, coeffs in enumerate(bx_tuple):
        if coeffs is not None:
            max_order = max(max_order, order)

    multipole_order = max_order + 1 if max_order >= 0 else 1
    if multipole_order > _MAX_MULTIPOLE_ORDER:
        raise ValueError(
            f"Unsupported multipole_order={multipole_order}; "
            f"max supported is {_MAX_MULTIPOLE_ORDER}"
        )

    bs_array = np.asarray(bs_spline.as_list(), dtype=float)
    by_array = np.zeros((multipole_order, _NUM_COEFFS), dtype=float)
    bx_array = np.zeros((multipole_order, _NUM_COEFFS), dtype=float)

    for order in range(multipole_order):
        if order < len(by_tuple) and by_tuple[order] is not None:
            by_array[order, :] = np.asarray(by_tuple[order].as_list(), dtype=float)

        if order < len(bx_tuple) and bx_tuple[order] is not None:
            bx_array[order, :] = np.asarray(bx_tuple[order].as_list(), dtype=float)

    return bs_array, by_array, bx_array, multipole_order


def _prepare_knl_ksl(knl=None, ksl=None):
    lengths = [1]
    if knl is not None:
        lengths.append(len(knl))
    if ksl is not None:
        lengths.append(len(ksl))

    target_len = max(lengths)
    knl_array = np.zeros(target_len, dtype=np.float64)
    ksl_array = np.zeros(target_len, dtype=np.float64)

    if knl is not None:
        knl_array[:len(knl)] = np.asarray(knl, dtype=np.float64)
    if ksl is not None:
        ksl_array[:len(ksl)] = np.asarray(ksl, dtype=np.float64)

    return knl_array, ksl_array


class SplineBoris(BeamElement):
    '''
    Thick element integrating the Lorentz force with a Boris stepper in a
    magnetic field represented by piecewise polynomials in the longitudinal
    coordinate.

    The field is expressed in a local longitudinal coordinate
    ``s_local \\in [0, length]``; any global ``s`` bookkeeping is handled at the
    lattice level.

    Parameters
    ----------
    bs : Spline4, optional
        Longitudinal field component as Hermite boundary data.
    bx : Spline4 or tuple/list of (Spline4 or None), optional
        Hermite data for the skew multipole components (Bx channel). A single
        ``Spline4`` corresponds to derivative order 0. A tuple/list item index
        gives the transverse derivative order with respect to ``x``;
        ``None`` entries are treated as zero.
    by : Spline4 or tuple/list of (Spline4 or None), optional
        Hermite data for the normal multipole components (By channel), with
        the same indexing semantics as ``bx``.
    length : float
        Physical length of the element in meters.
    n_steps : int
        Number of Boris substeps (must be ``>= 1``).
    shift_x : float, optional
        Horizontal offset of the field map in meters. Default is ``0``.
    shift_y : float, optional
        Vertical offset of the field map in meters. Default is ``0``.
    scale_b : float, optional
        Multiplicative scale factor applied to the magnetic field. Default is ``1``.
    radiation_flag : int, optional
        Radiation model flag. ``0`` disables radiation, non-zero values select
        synchrotron radiation models as for other thick elements.
    knl : array-like, optional
        Integrated strengths of additional normal multipole components in
        m**(-order). The corresponding kick is split over the Boris steps.
    ksl : array-like, optional
        Integrated strengths of additional skew multipole components in
        m**(-order). The corresponding kick is split over the Boris steps.

    Examples
    --------
    Build a one-meter element with a normal dipole field plus a normal
    quadrupole-gradient term and track particles through it:

    .. code-block:: python

        import xtrack as xt

        bs0 = xt.Spline4(
            val_start=0.02, der_start=0.0,
            val_end=0.02, der_end=0.0,
            mean=0.02,
        )
        bx0 = xt.Spline4(
            val_start=0.03, der_start=0.0,
            val_end=0.03, der_end=0.0,
            mean=0.03,
        )
        by0 = xt.Spline4(
            val_start=0.1, der_start=0.0,
            val_end=0.1, der_end=0.0,
            mean=0.1,
        )
        by1 = xt.Spline4(
            val_start=20.0, der_start=0.0,
            val_end=20.0, der_end=0.0,
            mean=20.0,
        )

        element = xt.SplineBoris(
            bs=bs0,
            by=(by0, by1),  # By = by0(s) + by1(s) * x + ...
            bx=(bx0,),      # Bx skew dipole component
            length=1.0,
            n_steps=100,
        )

        line = xt.Line(elements=[element])
        line.particle_ref = xt.Particles("electron", p0c=1e9)

        particles = line.particle_ref.copy()
        particles.x = 1e-3
        line.track(particles)

    Higher-order normal or skew components can be supplied by adding entries to
    ``by`` or ``bx``. The tuple index is the transverse derivative order with
    respect to ``x``: ``by=(by0, by1, by2)`` defines normal dipole,
    quadrupole-gradient and sextupole-like terms.
    '''

    isthick = True
    has_backtrack = True
    # Disable base transverse shifts - we use shift_x/shift_y as offsets in the field evaluation
    # Rotations should be apparent from the field map itself, not from transformation of the element.
    allow_rot_and_shift = False

    _SB_POLY_ORDER = _POLY_ORDER
    _SB_NUM_COEFFS = _NUM_COEFFS
    _SB_MAX_MULTIPOLE_ORDER = _MAX_MULTIPOLE_ORDER
    # Hermite field names used by FieldFitter/SplineBorisSequence helpers.
    _SB_HERMITE_SUFFIXES = _HERMITE_SUFFIXES

    _xofields = {
        'bs'                : xo.Float64[_SB_NUM_COEFFS],
        'knl'               : xo.Float64[:],
        'ksl'               : xo.Float64[:],
        'by'                : xo.Float64[:, _SB_NUM_COEFFS],
        'bx'                : xo.Float64[:, _SB_NUM_COEFFS],
        'multipole_order'   : xo.Int64,
        'length'            : xo.Float64,
        'n_steps'           : xo.Int64,
        'shift_x'           : xo.Field(xo.Float64, 0),  # Transverse shift in x [m] - used for field map offset
        'shift_y'           : xo.Field(xo.Float64, 0),  # Transverse shift in y [m] - used for field map offset
        'scale_b'           : xo.Field(xo.Float64, default=1),
        'radiation_flag'    : xo.Int64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/splineboris.h"',
    ]

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    def __init__(self,
                 bs=Spline4(0.0, 0.0, 0.0, 0.0, 0.0),
                 bx=Spline4(0.0, 0.0, 0.0, 0.0, 0.0),
                 by=Spline4(0.0, 0.0, 0.0, 0.0, 0.0),
                 length=1.0,
                 n_steps=1,
                 shift_x=0.0,
                 shift_y=0.0,
                 scale_b=1.0,
                 **kwargs,
    ):
        """Build the element from ``Spline4`` data and store Hermite boundary data in the xobject."""

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            super().__init__(**kwargs)
            return

        bs_array, by_array, bx_array, multipole_order = _validate_and_convert_to_array(bs, by, bx)

        if n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {n_steps}")
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f"length must be finite and > 0, got {length}")

        length = float(length)

        radiation_flag = kwargs.pop('radiation_flag', 0)
        knl = kwargs.pop('knl', None)
        ksl = kwargs.pop('ksl', None)
        knl, ksl = _prepare_knl_ksl(knl=knl, ksl=ksl)

        super().__init__(
            bs=bs_array,
            knl=knl,
            ksl=ksl,
            by=by_array,
            bx=bx_array,
            multipole_order=multipole_order,
            length=length,
            n_steps=n_steps,
            shift_x=shift_x,
            shift_y=shift_y,
            scale_b=scale_b,
            radiation_flag=radiation_flag,
            **kwargs,
        )

    def to_dict(self, copy_to_cpu=True):
        out = super().to_dict(copy_to_cpu=copy_to_cpu)

        bs_xo = out.pop('bs', None)
        by_xo = out.pop('by', None)
        bx_xo = out.pop('bx', None)

        if bs_xo is None:
            bs_xo = self._context.nparray_from_context_array(self.bs)
        if by_xo is None:
            by_xo = self._context.nparray_from_context_array(self.by)
        if bx_xo is None:
            bx_xo = self._context.nparray_from_context_array(self.bx)

        out['bs'] = Spline4(*bs_xo).as_dict()

        by = []
        bx = []
        for order in range(int(out['multipole_order'])):
            by_coeffs = np.asarray(by_xo[order], dtype=float)
            bx_coeffs = np.asarray(bx_xo[order], dtype=float)

            by.append(None if np.allclose(by_coeffs, 0, atol=1e-16)
                      else Spline4(*by_coeffs).as_dict())
            bx.append(None if np.allclose(bx_coeffs, 0, atol=1e-16)
                      else Spline4(*bx_coeffs).as_dict())

        out['by'] = by
        out['bx'] = bx

        out.pop('multipole_order', None)

        if 'knl' in out and np.allclose(out['knl'], 0, atol=1e-16):
            out.pop('knl', None)

        if 'ksl' in out and np.allclose(out['ksl'], 0, atol=1e-16):
            out.pop('ksl', None)

        return out

    @classmethod
    def from_dict(cls, dct, **kwargs):
        dct = dct.copy()

        dct.pop('__class__', None)
        dct.update(kwargs)
        return cls(**dct)

    def get_field(self, x, y, s_local):
        """Evaluate **B** in the element's local longitudinal coordinate.

        Parameters
        ----------
        x, y : float or array-like
            Transverse positions [m].
        s_local : float or array-like
            Local longitudinal coordinate(s) in the range ``[0, length]``.
            If array-like, it is broadcast together with ``x`` and ``y``.
        """
        x_arr, y_arr, s_loc = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(s_local, dtype=float),
        )

        outside = (s_loc < 0) | (s_loc > self.length)
        if np.any(outside):
            s_min = float(np.min(s_loc))
            s_max = float(np.max(s_loc))
            raise ValueError(
                "s_local contains values outside the local element range "
                f"[0, {self.length}] (min={s_min}, max={s_max})"
            )

        from .splineboris_src.spline_B_field_eval_python import evaluate_B

        # Build Python-side Hermite arrays from the xobject fields.
        bs = [self.bs[i] for i in range(self._SB_NUM_COEFFS)]

        by = []
        bx = []
        for order in range(self.multipole_order):
            by.append([self.by[order, j] for j in range(self._SB_NUM_COEFFS)])
            bx.append([self.bx[order, j] for j in range(self._SB_NUM_COEFFS)])

        bx_eval, by_eval, bs_eval = evaluate_B(
            x_arr - self.shift_x,
            y_arr - self.shift_y,
            s_loc,
            bs,
            by,
            bx,
            self.length,
            self.multipole_order,
        )

        bx_eval *= self.scale_b
        by_eval *= self.scale_b
        bs_eval *= self.scale_b

        if bx_eval.shape == ():
            return float(bx_eval), float(by_eval), float(bs_eval)
        return bx_eval, by_eval, bs_eval
