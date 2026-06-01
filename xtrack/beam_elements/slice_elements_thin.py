import xobjects as xo

from ..general import _pkg_root
from .slice_base import _SliceBase, COMMON_SLICE_XO_FIELDS, ID_RADIATION_FROM_PARENT
from .elements import (
    SynchrotronRadiationRecord, Quadrupole, Sextupole,
    Octupole, Bend, Multipole, DipoleEdge, RBend, MultipoleEdge, Marker,
    UniformSolenoid, Cavity, CrabCavity
)
from ..base_element import BeamElement

def _raise_if_parent_has_transverse_rotation(parent):
    if parent.rot_x_rad != 0 or parent.rot_y_rad != 0:
        raise ValueError(
            'Thin slice equivalent multipoles do not support parent '
            '`rot_x_rad` or `rot_y_rad` different from zero.')


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

        _raise_if_parent_has_transverse_rotation(self._parent)

        knl, ksl = self._parent.get_total_knl_ksl()
        knl = knl * self.weight
        ksl = ksl * self.weight

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
        '#include "xtrack/beam_elements/elements_src/thin_slice_sextupole.h"'
    ]

    def get_equivalent_element(self):

        _raise_if_parent_has_transverse_rotation(self._parent)

        knl, ksl = self._parent.get_total_knl_ksl()
        knl = knl * self.weight
        ksl = ksl * self.weight

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
        '#include "xtrack/beam_elements/elements_src/thin_slice_octupole.h"'
    ]

    def get_equivalent_element(self):

        _raise_if_parent_has_transverse_rotation(self._parent)

        knl, ksl = self._parent.get_total_knl_ksl()
        knl = knl * self.weight
        ksl = ksl * self.weight

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


class ThinSliceCavity(_ThinSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Cavity), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_cavity.h'),
    ]

    def get_equivalent_element(self):

        out = Cavity(length=0,
                     voltage=self._parent.voltage * self.weight,
                     frequency=self._parent.frequency,
                     harmonic=self._parent.harmonic,
                     lag=self._parent.lag,
                     phase=self._parent.phase,
                     lag_taper=self._parent.lag_taper,
                     absolute_time=self._parent.absolute_time,
                     _buffer=self._buffer)

        return out


class ThinSliceCrabCavity(_ThinSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(CrabCavity), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_crab_cavity.h'),
    ]

    def get_equivalent_element(self):

        out = CrabCavity(length=0,
                     crab_voltage=self._parent.crab_voltage * self.weight,
                     frequency=self._parent.frequency,
                     lag=self._parent.lag,
                     phase=self._parent.phase,
                     lag_taper=self._parent.lag_taper,
                     absolute_time=self._parent.absolute_time,
                     _buffer=self._buffer)

        return out


class ThinSliceBend(_ThinSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Bend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_bend.h"'
    ]

    def get_equivalent_element(self):

        _raise_if_parent_has_transverse_rotation(self._parent)

        knl, ksl = self._parent.get_total_knl_ksl()
        knl = knl * self.weight
        ksl = ksl * self.weight

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
        '#include "xtrack/beam_elements/elements_src/thin_slice_rbend.h"'
    ]

    def get_equivalent_element(self):

        _raise_if_parent_has_transverse_rotation(self._parent)

        if self._parent.rbend_model == "straight-body":
            return self # No replacement possible (not yet supported), element
                        # left where it is

        knl, ksl = self._parent.get_total_knl_ksl()
        knl = knl * self.weight
        ksl = ksl * self.weight

        length = self._parent.length * self.weight

        out = Multipole(knl=knl, ksl=ksl, length=length,
                        hxl=self._parent.h * length,
                        shift_x=self._parent.shift_x,
                        shift_y=self._parent.shift_y,
                        shift_s=self._parent.shift_s,
                        rot_s_rad=self._parent.rot_s_rad,
                        _buffer=self._buffer)
        return out


class ThinSliceMultipole(_ThinSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Multipole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/thin_slice_multipole.h"'
    ]

    def get_equivalent_element(self):
        _raise_if_parent_has_transverse_rotation(self._parent)

        knl, ksl = self._parent.get_total_knl_ksl()
        knl = knl * self.weight
        ksl = ksl * self.weight

        length = self._parent.length * self.weight

        if self.radiation_flag == ID_RADIATION_FROM_PARENT:
            radiation_flag = self._parent.radiation_flag
        else:
            radiation_flag = self.radiation_flag

        out = Multipole(knl=knl, ksl=ksl, length=length,
                        hxl=self._parent.hxl,
                        radiation_flag=radiation_flag,
                        delta_taper=self.delta_taper,
                        shift_x=self._parent.shift_x,
                        shift_y=self._parent.shift_y,
                        shift_s=self._parent.shift_s,
                        rot_s_rad=self._parent.rot_s_rad,
                        _buffer=self._buffer)
        return out
