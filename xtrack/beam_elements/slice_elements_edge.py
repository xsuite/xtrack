import xobjects as xo

from ..general import _pkg_root
from ..base_element import BeamElement
from .slice_elements_thin import _slice_copy, ID_RADIATION_FROM_PARENT, _common_xofields
from .elements import (
    Bend, Quadrupole, Sextupole,
    Octupole, RBend, UniformSolenoid, DipoleEdge, Marker, MultipoleEdge
)

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
        '#include <beam_elements/elements_src/thin_slice_bend_entry.h>'
    ]

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
        '#include <beam_elements/elements_src/thin_slice_bend_exit.h>'
    ]

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
        '#include <beam_elements/elements_src/thin_slice_quadrupole_entry.h>'
    ]

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
        '#include <beam_elements/elements_src/thin_slice_quadrupole_exit.h>'
    ]

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
        '#include <beam_elements/elements_src/thin_slice_sextupole_entry.h>'
    ]

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
        '#include <beam_elements/elements_src/thin_slice_sextupole_exit.h>'
    ]

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
        '#include <beam_elements/elements_src/thin_slice_octupole_entry.h>'
    ]

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
        '#include <beam_elements/elements_src/thin_slice_octupole_exit.h>'
    ]

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

class ThinSliceUniformSolenoidEntry(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(UniformSolenoid), **_common_xofields}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thin_slice_uniform_solenoid_entry.h>'
    ]

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
            raise NotImplementedError
        else:
            return Marker(_buffer=self._buffer)

class ThinSliceUniformSolenoidExit(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(UniformSolenoid), **_common_xofields}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thin_slice_uniform_solenoid_exit.h>'
    ]

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
            raise NotImplementedError
        else:
            return Marker(_buffer=self._buffer)


class ThinSliceRBendEntry(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(RBend), **_common_xofields}

    _extra_c_sources = [
        '#include <beam_elements/elements_src/thin_slice_rbend_entry.h>'
    ]

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
        '#include <beam_elements/elements_src/thin_slice_rbend_exit.h>'
    ]

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

