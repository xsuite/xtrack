import xobjects as xo

from ..base_element import BeamElement
from .slice_base import _SliceBase, COMMON_SLICE_XO_FIELDS
from .bend import Bend
from .cavity import Cavity
from .crab_cavity import CrabCavity
from .multipole import Multipole
from .octupole import Octupole
from .quadrupole import Quadrupole
from .rbend import RBend
from .sextupole import Sextupole
from .solenoid import Solenoid
from .uniform_solenoid import UniformSolenoid

class _ThickSliceElementBase(_SliceBase):

    rot_and_shift_from_parent = True
    allow_loss_refinement = True
    isthick = True
    _inherit_strengths = True

class ThickSliceBend(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Bend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_bend.h"'
    ]

class ThickSliceRBend(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(RBend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_rbend.h"'
    ]

    def track_frame(self, frame, backtrack=False):

        if self._parent.rbend_model == "straight-body":
            ll = self._parent.length_straight * self.weight
            aa = 0
        else:
            ll = self._parent.length * self.weight
            aa = self._parent.angle * self.weight

        if backtrack:
            ll *= -1
            aa *= -1

        frame.arc(length=ll, angle=aa, tilt=self._parent.rot_s_rad)


class ThickSliceQuadrupole(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Quadrupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_quadrupole.h"'
    ]


class ThickSliceSextupole(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Sextupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_sextupole.h"'
    ]

class ThickSliceOctupole(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Octupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_octupole.h"'
    ]

class ThickSliceCavity(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Cavity), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_cavity.h"'
    ]

class ThickSliceCrabCavity(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(CrabCavity), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_crab_cavity.h"'
    ]

class ThickSliceMultipole(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Multipole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_multipole.h"'
    ]

class ThickSliceUniformSolenoid(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(UniformSolenoid), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_uniform_solenoid.h"'
    ]

class ThickSliceSolenoid(_ThickSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Solenoid), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_solenoid.h"'
    ]
