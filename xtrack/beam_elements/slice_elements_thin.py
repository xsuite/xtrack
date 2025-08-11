import xobjects as xo

from ..general import _pkg_root
from .slice_base import _SliceBase, COMMON_SLICE_XO_FIELDS, ID_RADIATION_FROM_PARENT
from .elements import (
    SynchrotronRadiationRecord, Quadrupole, Sextupole,
    Octupole, Bend, Multipole, DipoleEdge, RBend, MultipoleEdge, Marker,
    UniformSolenoid
)
from ..base_element import BeamElement

class _ThinSliceElementBase(_SliceBase):

    rot_and_shift_from_parent = True
    allow_loss_refinement = False
    isthick = False
    _inherit_strengths = True


class ThinSliceQuadrupole(_ThinSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Quadrupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_quadrupole.h'),
    ]

    def get_equivalent_element(self):

        knl = self._parent.knl.copy() * self.weight
        ksl = self._parent.ksl.copy() * self.weight

        knl[1] += self._parent.k1 * self._parent.length * self.weight
        ksl[1] += self._parent.k1s * self._parent.length * self.weight

        length = self._parent.length * self.weight

        if self.radiation_flag == ID_RADIATION_FROM_PARENT:
            radiation_flag = self._parent.radiation_flag
        else:
            radiation_flag = self.radiation_flag

        out = Multipole(knl=knl, ksl=ksl, length=length,
                        hxl=0,
                        radiation_flag=radiation_flag,
                        delta_taper=self.delta_taper,
                        shift_x=self._parent.shift_x,
                        shift_y=self._parent.shift_y,
                        shift_s=self._parent.shift_s,
                        rot_s_rad=self._parent.rot_s_rad,
                        _buffer=self._buffer)
        return out


class ThinSliceSextupole(_ThinSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Sextupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thin_slice_sextupole.h>'
    ]

    def get_equivalent_element(self):

        knl = self._parent.knl.copy() * self.weight
        ksl = self._parent.ksl.copy() * self.weight

        knl[2] += self._parent.k2 * self._parent.length * self.weight
        ksl[2] += self._parent.k2s * self._parent.length * self.weight

        length = self._parent.length * self.weight

        if self.radiation_flag == ID_RADIATION_FROM_PARENT:
            radiation_flag = self._parent.radiation_flag
        else:
            radiation_flag = self.radiation_flag

        out = Multipole(knl=knl, ksl=ksl, length=length,
                        hxl=0,
                        radiation_flag=radiation_flag,
                        delta_taper=self.delta_taper,
                        shift_x=self._parent.shift_x,
                        shift_y=self._parent.shift_y,
                        shift_s=self._parent.shift_s,
                        rot_s_rad=self._parent.rot_s_rad,
                        _buffer=self._buffer)
        return out


class ThinSliceOctupole(_ThinSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Octupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thin_slice_octupole.h>'
    ]

    def get_equivalent_element(self):

        knl = self._parent.knl.copy() * self.weight
        ksl = self._parent.ksl.copy() * self.weight

        knl[3] += self._parent.k3 * self._parent.length * self.weight
        ksl[3] += self._parent.k3s * self._parent.length * self.weight

        length = self._parent.length * self.weight

        if self.radiation_flag == ID_RADIATION_FROM_PARENT:
            radiation_flag = self._parent.radiation_flag
        else:
            radiation_flag = self.radiation_flag

        out = Multipole(knl=knl, ksl=ksl, length=length,
                        hxl=0,
                        radiation_flag=radiation_flag,
                        delta_taper=self.delta_taper,
                        shift_x=self._parent.shift_x,
                        shift_y=self._parent.shift_y,
                        shift_s=self._parent.shift_s,
                        rot_s_rad=self._parent.rot_s_rad,
                        _buffer=self._buffer)
        return out


class ThinSliceBend(_ThinSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Bend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thin_slice_bend.h>'
    ]

    def get_equivalent_element(self):
        knl = self._parent.knl.copy() * self.weight
        ksl = self._parent.ksl.copy() * self.weight

        knl[0] += self._parent.k0 * self._parent.length * self.weight
        knl[1] += self._parent.k1 * self._parent.length * self.weight

        length = self._parent.length * self.weight

        if self.radiation_flag == ID_RADIATION_FROM_PARENT:
            radiation_flag = self._parent.radiation_flag
        else:
            radiation_flag = self.radiation_flag

        out = Multipole(knl=knl, ksl=ksl, length=length,
                        hxl=self._parent.h * length,
                        radiation_flag=radiation_flag,
                        delta_taper=self.delta_taper,
                        shift_x=self._parent.shift_x,
                        shift_y=self._parent.shift_y,
                        shift_s=self._parent.shift_s,
                        rot_s_rad=self._parent.rot_s_rad,
                        _buffer=self._buffer)
        return out

class ThinSliceRBend(_ThinSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(RBend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thin_slice_rbend.h>'
    ]

    def get_equivalent_element(self):
        knl = self._parent.knl.copy() * self.weight
        ksl = self._parent.ksl.copy() * self.weight

        knl[0] += self._parent.k0 * self._parent.length * self.weight
        knl[1] += self._parent.k1 * self._parent.length * self.weight

        length = self._parent.length * self.weight

        out = Multipole(knl=knl, ksl=ksl, length=length,
                        hxl=self._parent.h * length,
                        shift_x=self._parent.shift_x,
                        shift_y=self._parent.shift_y,
                        shift_s=self._parent.shift_s,
                        rot_s_rad=self._parent.rot_s_rad,
                        _buffer=self._buffer)
        return out
