import xobjects as xo

from ..general import _pkg_root
from ..base_element import BeamElement
from .slice_base import _SliceBase, COMMON_SLICE_XO_FIELDS
from .elements import (
    SynchrotronRadiationRecord, Bend, Quadrupole, Sextupole,
    Octupole, Solenoid, Drift, RBend, UniformSolenoid
)
from ..survey import advance_element as survey_advance_element

class _DriftSliceElementBase(_SliceBase):

    rot_and_shift_from_parent = False
    allow_loss_refinement = True
    isthick=True
    _inherit_strengths = False

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
