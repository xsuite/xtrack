import xobjects as xo

from ..general import _pkg_root
from ..random import RandomUniformAccurate, RandomExponential
from .elements import (
    SynchrotronRadiationRecord, Quadrupole, Sextupole,
    Octupole, Bend, Multipole, DipoleEdge, RBend,
)
from ..base_element import BeamElement

xo.context_default.kernels.clear()

_common_xofields = {
    'radiation_flag': xo.Int64,
    'delta_taper': xo.Float64,
    'weight': xo.Float64,
}
_common_c_sources = [
    _pkg_root.joinpath('headers/constants.h'),
    _pkg_root.joinpath('headers/synrad_spectrum.h'),
    _pkg_root.joinpath('beam_elements/elements_src/track_multipole.h')
]

def _slice_copy(self, **kwargs):
    out = BeamElement.copy(self, **kwargs)
    out._parent = None
    out.parent_name = self.parent_name
    return out


class ThinSliceQuadrupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = True

    _xofields = {'_parent': xo.Ref(Quadrupole), **_common_xofields}

    _extra_c_sources = _common_c_sources + [
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_quadrupole.h')]

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

    def get_equivalent_element(self):

        knl = self._parent.knl.copy() * self.weight
        ksl = self._parent.ksl.copy() * self.weight

        knl[1] += self._parent.k1 * self._parent.length * self.weight
        ksl[1] += self._parent.k1s * self._parent.length * self.weight

        length = self._parent.length * self.weight

        out = Multipole(knl=knl, ksl=ksl, length=length,
                        hxl=0,
                        shift_x=self._parent.shift_x,
                        shift_y=self._parent.shift_y,
                        shift_s=self._parent.shift_s,
                        rot_s_rad=self._parent.rot_s_rad,
                        _buffer=self._buffer)
        return out


class ThinSliceSextupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = True

    _xofields = {'_parent': xo.Ref(Sextupole), **_common_xofields}

    _extra_c_sources = _common_c_sources + [
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_sextupole.h')]

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

    def get_equivalent_element(self):

        knl = self._parent.knl.copy() * self.weight
        ksl = self._parent.ksl.copy() * self.weight

        knl[2] += self._parent.k2 * self._parent.length * self.weight
        ksl[2] += self._parent.k2s * self._parent.length * self.weight

        length = self._parent.length * self.weight

        out = Multipole(knl=knl, ksl=ksl, length=length,
                        hxl=0,
                        shift_x=self._parent.shift_x,
                        shift_y=self._parent.shift_y,
                        shift_s=self._parent.shift_s,
                        rot_s_rad=self._parent.rot_s_rad,
                        _buffer=self._buffer)
        return out


class ThinSliceOctupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = True

    _xofields = {'_parent': xo.Ref(Octupole), **_common_xofields}

    _extra_c_sources = _common_c_sources + [
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_octupole.h')]

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

    def get_equivalent_element(self):

        knl = self._parent.knl.copy() * self.weight
        ksl = self._parent.ksl.copy() * self.weight

        knl[3] += self._parent.k3 * self._parent.length * self.weight
        ksl[3] += self._parent.k3s * self._parent.length * self.weight

        length = self._parent.length * self.weight

        out = Multipole(knl=knl, ksl=ksl, length=length,
                        hxl=0,
                        shift_x=self._parent.shift_x,
                        shift_y=self._parent.shift_y,
                        shift_s=self._parent.shift_s,
                        rot_s_rad=self._parent.rot_s_rad,
                        _buffer=self._buffer)
        return out


class ThinSliceBend(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = True

    _xofields = {'_parent': xo.Ref(Bend), **_common_xofields}

    _extra_c_sources = _common_c_sources + [
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_bend.h')]

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


class ThinSliceBendEntry(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Bend), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_bend_entry.h')]

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

    def get_equivalent_element(self):

        return DipoleEdge(
            k=self._parent.k0,
            e1=self._parent.edge_entry_angle,
            e1_fd=self._parent.edge_entry_angle_fdown,
            hgap=self._parent.edge_entry_hgap,
            fint=self._parent.edge_entry_fint,
            model=self._parent.edge_entry_model,
            side='entry',
            _buffer=self._buffer
        )


class ThinSliceBendExit(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Bend), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_bend_exit.h')]

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

    def get_equivalent_element(self):

            return DipoleEdge(
                k=self._parent.k0,
                e1=self._parent.edge_exit_angle,
                e1_fd=self._parent.edge_exit_angle_fdown,
                hgap=self._parent.edge_exit_hgap,
                fint=self._parent.edge_exit_fint,
                model=self._parent.edge_exit_model,
                side='exit',
                _buffer=self._buffer
            )


class ThinSliceRBend(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = True

    _xofields = {'_parent': xo.Ref(RBend), **_common_xofields}

    _extra_c_sources = _common_c_sources + [
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_rbend.h')]

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


class ThinSliceRBendEntry(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(RBend), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_rbend_entry.h')]

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

    def get_equivalent_element(self):

        return DipoleEdge(
            k=self._parent.k0,
            e1=self._parent.edge_entry_angle + self._parent.angle / 2,
            e1_fd=self._parent.edge_entry_angle_fdown,
            hgap=self._parent.edge_entry_hgap,
            fint=self._parent.edge_entry_fint,
            model=self._parent.edge_entry_model,
            side='entry',
            _buffer=self._buffer
        )


class ThinSliceRBendExit(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(RBend), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_rbend_exit.h')]

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

    def get_equivalent_element(self):

            return DipoleEdge(
                k=self._parent.k0,
                e1=self._parent.edge_exit_angle + self._parent.angle / 2,
                e1_fd=self._parent.edge_exit_angle_fdown,
                hgap=self._parent.edge_exit_hgap,
                fint=self._parent.edge_exit_fint,
                model=self._parent.edge_exit_model,
                side='exit',
                _buffer=self._buffer
            )
