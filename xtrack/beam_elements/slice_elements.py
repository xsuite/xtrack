import xobjects as xo

from ..general import _pkg_root
from ..random import RandomUniformAccurate, RandomExponential
from .elements import (
    SynchrotronRadiationRecord, Quadrupole, Sextupole,
    Octupole, Bend, Multipole, DipoleEdge, RBend, MultipoleEdge, Marker
)
from ..base_element import BeamElement

ID_RADIATION_FROM_PARENT = 10

xo.context_default.kernels.clear()

_common_xofields = {
    'radiation_flag': xo.Field(xo.Int64, default=ID_RADIATION_FROM_PARENT),
    'delta_taper': xo.Float64,
    'weight': xo.Float64,
}

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

    _extra_c_sources = [
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

    _extra_c_sources = [
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

    _extra_c_sources = [
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

    _extra_c_sources = [
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
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_nonlinear.h'),
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

        if self._parent.edge_entry_active:
            return DipoleEdge(
                k=self._parent.k0,
                e1=self._parent.edge_entry_angle,
                e1_fd=self._parent.edge_entry_angle_fdown,
                hgap=self._parent.edge_entry_hgap,
                fint=self._parent.edge_entry_fint,
                model=self._parent.edge_entry_model,
                side='entry',
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)


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
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_nonlinear.h'),
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

        if self._parent.edge_exit_active:
            return DipoleEdge(
                k=self._parent.k0,
                e1=self._parent.edge_exit_angle,
                e1_fd=self._parent.edge_exit_angle_fdown,
                hgap=self._parent.edge_exit_hgap,
                fint=self._parent.edge_exit_fint,
                model=self._parent.edge_exit_model,
                delta_taper=self.delta_taper,
                side='exit',
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)


class ThinSliceQuadrupoleEntry(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Quadrupole), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_quadrupole_entry.h')]

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
        if self._parent.edge_entry_active:
            return MultipoleEdge(
                kn=[0, self._parent.k1],
                ks=[0, self._parent.k1s],
                is_exit=False,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)


class ThinSliceQuadrupoleExit(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Quadrupole), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_quadrupole_exit.h')]

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
        if self._parent.edge_exit_active:
            return MultipoleEdge(
                kn=[0, self._parent.k1],
                ks=[0, self._parent.k1s],
                is_exit=True,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)


class ThinSliceSextupoleEntry(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Sextupole), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_sextupole_entry.h')]

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
        if self._parent.edge_entry_active:
            return MultipoleEdge(
                kn=[0, 0, self._parent.k2],
                ks=[0, 0, self._parent.k2s],
                is_exit=False,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)


class ThinSliceSextupoleExit(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Sextupole), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_sextupole_exit.h')]

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
        if self._parent.edge_exit_active:
            return MultipoleEdge(
                kn=[0, 0, self._parent.k2],
                ks=[0, 0, self._parent.k2s],
                is_exit=True,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)


class ThinSliceOctupoleEntry(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Octupole), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_octupole_entry.h')]

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
        if self._parent.edge_entry_active:
            return MultipoleEdge(
                kn=[0, 0, 0, self._parent.k3],
                ks=[0, 0, 0, self._parent.k3s],
                is_exit=False,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)


class ThinSliceOctupoleExit(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Octupole), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_octupole_exit.h')]

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
        if self._parent.edge_exit_active:
            return MultipoleEdge(
                kn=[0, 0, 0, self._parent.k3],
                ks=[0, 0, 0, self._parent.k3s],
                is_exit=True,
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)


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

    _extra_c_sources = [
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
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_nonlinear.h'),
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

        if self._parent.edge_entry_active:
            return DipoleEdge(
                k=self._parent.k0,
                e1=self._parent.edge_entry_angle + self._parent.angle / 2,
                e1_fd=self._parent.edge_entry_angle_fdown,
                hgap=self._parent.edge_entry_hgap,
                fint=self._parent.edge_entry_fint,
                model=self._parent.edge_entry_model,
                side='entry',
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)


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
        _pkg_root.joinpath('beam_elements/elements_src/track_dipole_edge_nonlinear.h'),
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

        if self._parent.edge_exit_active:
            return DipoleEdge(
                k=self._parent.k0,
                e1=self._parent.edge_exit_angle + self._parent.angle / 2,
                e1_fd=self._parent.edge_exit_angle_fdown,
                hgap=self._parent.edge_exit_hgap,
                fint=self._parent.edge_exit_fint,
                model=self._parent.edge_exit_model,
                side='exit',
                shift_x=self._parent.shift_x,
                shift_y=self._parent.shift_y,
                shift_s=self._parent.shift_s,
                rot_s_rad=self._parent.rot_s_rad,
                _buffer=self._buffer
            )
        else:
            return Marker(_buffer=self._buffer)

