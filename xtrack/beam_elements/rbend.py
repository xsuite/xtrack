# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import numpy as np
import xobjects as xo
import xtrack as xt
from ..random import (
    RandomExponential,
    RandomUniformAccurate,
)
from ._common import (
    SynchrotronRadiationRecord,
    _BendCommon,
    _HasIntegrator,
    _HasKnlKsl,
    _HasModelCurved,
    _INDEX_TO_RBEND_MODEL,
    _NOEXPR_FIELDS,
    _RBEND_MODEL_TO_INDEX,
    _docstring_general_notes,
    _for_docstring_alignment,
    _for_docstring_edge_bend,
)

class RBend(_BendCommon, BeamElement):
    _docstring_start = \
    """Rectangular bending magnet element.

    Parameters
    ----------
    length_strait : float
        Length of the element in meters along the axis of the magnet (straight line
        between entry and exit points). This is different from the length of the
        reference trajectory, i.e. the increase of the `s` coordinate through the
        element, which is computed internally and can be inspected via the
        `length` property.
    angle : float
        Angle of the bend in radians. This is the angle by which the reference
        trajectory is bent in the horizontal plane.
    k0 : float
        Strength of the horizontal dipolar component in units of m^-1.
        It can be set to the string value 'from_h', in which case `k0` is
        computed from the curvature defined by `angle` and `length`
        (i.e. `k0 = h = angle/length`) and `k0_from_h` is set to True.
    k1 : float
        Strength of the quadrupolar component in units of m^-2.
    k2 : float
        Strength of the sextupolar component in units of m^-3.
    k0_from_h : bool
        If True, `k0` is computed from the curvature defined by `angle` and
        `length` (i.e. `k0 = h = angle/length`). Default is True. The flag
        becomes false when `k0` is set directly to a numeric value.
    rbend_model : str
        Model used for the rectangular bend. Possible values are:
        "adaptive', "curved-body", "straight-body". Default is "adaptive',
        which falls back to "curved-body".
    rbend_angle_diff : float
        Difference in radians between the angle of the reference trajectory
        with respect to the magnet axis at the entrance and exit of the magnet.
        See drawing on Xsuite Physics Guide. Default is 0.0.
    rbend_shift : float
        Shift of the magnet body, in meters, defined as the displacement
        of the reference trajectory with respect to the magnet axis at the center
        of the magnet. This parameter has effect only when `rbend_model` is
        "straight-body". Default is 0.0.
    rbend_compensate_sagitta : bool
        If True, the magnet body is shifted by half of the trajectory sagitta,
        defined as (1 / h) * (1 - cos(angle / 2)). The shift is added to `rbend_shift`.
        This parameter has effect only when `rbend_model` is "straight-body".
        Default is True.
    """.strip()

    _docstring_knl_rel_ksl_rel = \
    """knl_rel : array
        Relative integrated strength of the normal components with respect to the
        main component k0. The effect of knl_rel is added to the one of knl.
    ksl_rel : array
        Relative integrated strength of the skew components with respect to the
        main component k0. The effect of ksl_rel is added to the one of ksl.
    """.strip()

    __doc__ = '\n    '.join([_docstring_start, _HasKnlKsl._for_docstring,
            _docstring_knl_rel_ksl_rel,
            _HasModelCurved._for_docstring, _HasIntegrator._for_docstring,
            _for_docstring_edge_bend, _for_docstring_alignment, '\n',
            _docstring_general_notes, '\n\n'])

    _xofields = {
        **_BendCommon._common_xofields,
        'length_straight': xo.Float64,
        'rbend_model': xo.Int64,
        'rbend_compensate_sagitta': xo.Field(xo.Int64, default=True),
        'rbend_shift': xo.Float64,
        'rbend_angle_diff': xo.Float64,
    }

    allow_loss_refinement = True

    _rename = {
        **_BendCommon._common_rename,
        'length_straight': '_length_straight',
        'rbend_model': '_rbend_model',
        'rbend_angle_diff': '_rbend_angle_diff',
        'rbend_compensate_sagitta': '_rbend_compensate_sagitta',
    }

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/rbend.h"',
    ]

    _noexpr_fields = _NOEXPR_FIELDS

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if 'h' in kwargs:
            raise ValueError("Setting `h` directly is not allowed. "
                                "Set `length` and `angle` instead.")

        if 'length' in kwargs:
            raise ValueError("Setting `length` directly is not allowed for RBend. "
                             "Set `length_straight` instead.")

        to_be_set_with_properties = []
        for nn in ['length_straight', 'angle', 'k0_from_h', 'edge_entry_model',
                   'edge_exit_model', 'rbend_angle_diff', 'rbend_model', 'k0']:
            if nn in kwargs:
                to_be_set_with_properties.append((nn, kwargs.pop(nn)))

        _HasKnlKsl.__init__(self, **kwargs) # Handles knl, ksl, order, model, integrator

        for nn, val in to_be_set_with_properties:
            setattr(self, nn, val)

    @classmethod
    def from_dict(cls, dct, **kwargs):

        dct = dct.copy()

        if 'length' in dct:
            assert 'length_straight' in dct
            dct.pop('length')

        return super().from_dict(dct, **kwargs)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        raise RuntimeError("Setting `length` directly is not allowed for RBend. "
                           "Set `length_straight` instead.")

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        self._update_rbend_h_length_k0()

    @property
    def rbend_angle_diff(self):
        return self._rbend_angle_diff

    @rbend_angle_diff.setter
    def rbend_angle_diff(self, value):
        self._rbend_angle_diff = value
        self._update_rbend_h_length_k0()

    @property
    def length_straight(self):
        return self._length_straight

    @length_straight.setter
    def length_straight(self, value):
        self._length_straight = value
        self._update_rbend_h_length_k0()

    def _update_rbend_h_length_k0(self):
        _angle = self._angle
        _length_straight = self._length_straight
        _rbend_angle_diff = self._rbend_angle_diff

        theta_in = 0.5 * _angle - _rbend_angle_diff / 2
        theta_out = 0.5 * _angle + _rbend_angle_diff / 2
        if abs(_angle) < 1e-10:
            length = _length_straight
            h = 0
        elif abs(_length_straight) < 1e-10:
            length = 0.0
            h = 0
        else:
            h = (np.sin(theta_in) + np.sin(theta_out)) / _length_straight
            length = _angle / h

        self._h = h
        self._angle = _angle
        self._length = length
        if self.k0_from_h:
            self._k0 = self._h

    @property
    def rbend_model(self):
        return _INDEX_TO_RBEND_MODEL[self._rbend_model]

    @rbend_model.setter
    def rbend_model(self, value):
        try:
            self._rbend_model = _RBEND_MODEL_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid rbend_model: {value}')

    @property
    def rbend_compensate_sagitta(self):
        return bool(self._rbend_compensate_sagitta)

    @rbend_compensate_sagitta.setter
    def rbend_compensate_sagitta(self, value):
        self._rbend_compensate_sagitta = int(bool(value))

    @property
    def hxl(self): return self.h * self.length

    @property
    def _angle_in(self):
        return 0.5 * self.angle - self._rbend_angle_diff / 2

    @property
    def _angle_out(self):
        return 0.5 * self.angle + self._rbend_angle_diff / 2

    @property
    def _x0_mid(self):
        out = -self.rbend_shift
        if abs(self.angle) > 1e-10 and self.rbend_compensate_sagitta:
            # 1 - cos(u) = 2 * sin(u / 2)**2, to avoid the cancellation.
            out += np.sin(self.angle / 4) ** 2 / self.h
        return out

    @property
    def _x0_in(self):
        # Rationalised form of (1 / h) * (sqrt_mid - cos_theta_in), in which h
        # cancels exactly. See `track_magnet.h` for the derivation.
        out = self._x0_mid
        if abs(self.angle) > 1e-10:
            px0_in = np.sin(self._angle_in)
            px0_mid = px0_in - self.h * self.length_straight / 2
            sqrt_mid = np.sqrt(1 - px0_mid * px0_mid)
            cos_theta_in = np.cos(self._angle_in)
            out -= (0.5 * self.length_straight * (px0_in + px0_mid)
                    / (sqrt_mid + cos_theta_in))
        return out

    @property
    def _x0_out(self):
        out = self._x0_mid
        if abs(self.angle) > 1e-10:
            px0_out = np.sin(self._angle_out)
            px0_mid = px0_out - self.h * self.length_straight / 2
            sqrt_mid = np.sqrt(1 - px0_mid * px0_mid)
            cos_theta_out = np.cos(self._angle_out)
            out -= (0.5 * self.length_straight * (px0_out + px0_mid)
                    / (cos_theta_out + sqrt_mid))
        return out

    @property
    def radiation_flag(self): return 0.0

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceRBend

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceRBend

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceRBend

    @property
    def _entry_slice_class(self):
        return xt.ThinSliceRBendEntry

    @property
    def _exit_slice_class(self):
        return xt.ThinSliceRBendExit

    @property
    def _repr_fields(self):
        return ['length_straight', 'angle'] + super()._repr_fields

    def to_dict(self, copy_to_cpu=True):
        out = super().to_dict(copy_to_cpu=copy_to_cpu)

        for kk in {'angle', 'length_straight'}:
            if f'_{kk}' in out:
                out.pop(f'_{kk}')
            out[kk] = getattr(self, kk)

        return out
