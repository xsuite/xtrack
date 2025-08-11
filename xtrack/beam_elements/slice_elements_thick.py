import xobjects as xo

from ..base_element import BeamElement
from .slice_base import _SliceBase, COMMON_SLICE_XO_FIELDS
from .elements import (
    SynchrotronRadiationRecord, Bend, Quadrupole, Sextupole,
    Octupole, Solenoid, Drift, RBend, UniformSolenoid
)
from ..survey import advance_element as survey_advance_element

class _ThickSliceElementBase(_SliceBase):

    rot_and_shift_from_parent = True
    allow_loss_refinement = True
    isthick = True
    _inherit_strengths = True

class ThickSliceBend(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Bend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thick_slice_bend.h>'
    ]

class ThickSliceRBend(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(RBend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thick_slice_rbend.h>'
    ]

    def _propagate_survey(self, v, w, backtrack):

        if self._parent.rbend_model == "straight-body":
            ll = self._parent.length_straight * self.weight
            aa = 0
        else:
            ll = self._parent.length * self.weight
            aa = self._parent.angle * self.weight

        if backtrack:
            ll *= -1
            aa *= -1

        v, w = survey_advance_element(
            v               = v,
            w               = w,
            length          = ll,
            angle           = aa,
            tilt            = self._parent.rot_s_rad,
            ref_shift_x     = 0,
            ref_shift_y     = 0,
            ref_rot_x_rad   = 0,
            ref_rot_y_rad   = 0,
            ref_rot_s_rad   = 0,
        )

        return v, w


class ThickSliceQuadrupole(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Quadrupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thick_slice_quadrupole.h>'
    ]


class ThickSliceSextupole(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Sextupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thick_slice_sextupole.h>'
    ]

class ThickSliceOctupole(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Octupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thick_slice_octupole.h>'
    ]

class ThickSliceUniformSolenoid(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(UniformSolenoid), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thick_slice_uniform_solenoid.h>'
    ]

class ThickSliceSolenoid(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Solenoid), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thick_slice_solenoid.h>'
    ]
