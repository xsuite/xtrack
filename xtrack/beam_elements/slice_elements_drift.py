import xobjects as xo

from ..general import _pkg_root
from ..base_element import BeamElement
from .elements import (
    SynchrotronRadiationRecord, Bend, Quadrupole, Sextupole,
    Octupole, Solenoid, Drift, RBend, UniformSolenoid
)
from ..random import RandomUniformAccurate, RandomExponential
from ..survey import advance_element as survey_advance_element

from .slice_elements_thin import _slice_copy, ID_RADIATION_FROM_PARENT


COMMON_SLICE_XO_FIELDS = {
    'radiation_flag': xo.Field(xo.Int64, default=ID_RADIATION_FROM_PARENT),
    'delta_taper': xo.Float64,
    'weight': xo.Float64,
}

class _DriftSliceElementBase:

    allow_rot_and_shift = False
    allow_loss_refinement = True
    rot_and_shift_from_parent = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = False

    copy = _slice_copy

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['parent_name'] = self.parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj.parent_name = dct['parent_name']
        return obj


class DriftSliceBend(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Bend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/drift_slice_bend.h>'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out


class DriftSliceRBend(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(RBend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/drift_slice_rbend.h>'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out

    def _propagate_survey(self, v, w, backtrack):

        if self._parent.rbend_model == "straight-body":
            ll = self._parent.length_straight * self.weight
        else:
            ll = self._parent.length * self.weight

        if backtrack:
            ll *= -1

        v, w = survey_advance_element(
            v               = v,
            w               = w,
            length          = ll,
            angle           = 0,
            tilt            = 0,
            ref_shift_x     = 0,
            ref_shift_y     = 0,
            ref_rot_x_rad   = 0,
            ref_rot_y_rad   = 0,
            ref_rot_s_rad   = 0,
        )

        return v, w


class DriftSliceQuadrupole(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Quadrupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/drift_slice_quadrupole.h>'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out


class DriftSliceSextupole(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Sextupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/drift_slice_sextupole.h>'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out


class DriftSliceOctupole(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Octupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/drift_slice_octupole.h>'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out


class DriftSlice(_DriftSliceElementBase, BeamElement):
    _xofields = {'_parent': xo.Ref(Drift), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/drift_slice.h>'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out