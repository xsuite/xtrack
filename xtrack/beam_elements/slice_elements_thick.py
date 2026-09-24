import numpy as np
import xobjects as xo

from ..base_element import BeamElement
from .slice_base import _SliceBase, COMMON_SLICE_XO_FIELDS
from .bend import Bend
from .bfield_expansion import BFieldExpansion
from .cavity import Cavity
from .crab_cavity import CrabCavity
from .device import Device
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


class ThickSliceBFieldExpansion(_ThickSliceElementBase, BeamElement):
    """A longitudinal interval sharing a BFieldExpansion's coefficients/cache.

    ``slice_offset`` is measured from the parent's entrance. Its stored fraction
    follows changes of the parent's length, as does the slice's weight.
    ``s_start`` includes the parent's polynomial-coordinate origin.
    """

    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    allow_loss_refinement = False
    # The profile is nonuniform: compute strengths on this interval on demand.
    _inherit_strengths = False
    _line_attr_properties = ('hxl', 'knl', 'ksl', 'ksoll')

    _xofields = {
        '_parent': xo.Ref(BFieldExpansion),
        **COMMON_SLICE_XO_FIELDS,
        '_slice_offset_fraction': xo.Float64,
    }
    del _xofields['slice_offset']

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_bfieldexpansion.h"'
    ]

    def __init__(self, slice_offset=None, **kwargs):
        if slice_offset is not None:
            parent = kwargs['_parent']
            kwargs['_slice_offset_fraction'] = slice_offset / parent.length
        super().__init__(**kwargs)

    @property
    def slice_offset(self):
        return self._slice_offset_fraction * self._parent.length

    @property
    def s_start(self):
        return self._parent.s_start + self.slice_offset

    @property
    def nstep(self):
        return max(1, int(np.ceil(self._parent.nstep * self.weight)))

    @property
    def angle(self):
        return self._parent.angle * self.weight

    @property
    def hxl(self):
        return self.angle

    def _integrated_strength(self, name):
        result = self._parent._integrate_coefficients(
            name, self.s_start, self._parent.length * self.weight)
        result.flags.writeable = False
        return result

    @property
    def knl(self):
        return self._integrated_strength('knc')

    @property
    def ksl(self):
        return self._integrated_strength('ksc')

    @property
    def ksoll(self):
        return self._integrated_strength('ksol')

    def get_field(self, x, y, s_local):
        """Evaluate the field at distances s_local from this slice's entrance."""
        return self._parent.get_field(x, y, np.asarray(s_local) + self.slice_offset)

    def track(self, particles=None, increment_at_element=False):
        self._parent._check_spin_tracking(particles)
        return super().track(particles, increment_at_element=increment_at_element)


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

class ThickSliceDevice(_ThickSliceElementBase, BeamElement):

    # A device has no strengths to inherit, but it does inherit the parent's
    # misalignment through `rot_and_shift_from_parent`, which is why this is
    # not a `_DriftSliceElementBase`: that one switches the transformations off
    # and would silently drop the misalignment of a sliced device.
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Device), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thick_slice_device.h"'
    ]

    def get_equivalent_element(self):
        return Device(length=self._parent.length * self.weight,
                      model=self._parent.model,
                      _buffer=self._buffer)
