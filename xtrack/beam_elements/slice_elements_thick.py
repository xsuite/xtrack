import xobjects as xo

from ..general import _pkg_root
from ..base_element import BeamElement
from .elements import (
    SynchrotronRadiationRecord, Bend, Quadrupole, Sextupole,
    Octupole, Solenoid, Drift, RBend,
)
from ..random import RandomUniformAccurate, RandomExponential

from .slice_elements import _slice_copy

_common_xofields = {
    'weight': xo.Float64,
}


class ThickSliceBend(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = True

    _xofields = {'_parent': xo.Ref(Bend), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_multipolar_components.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_bend.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_cfd.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_bend.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thick_slice_bend.h')]

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


class ThickSliceRBend(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = True

    _xofields = {'_parent': xo.Ref(RBend), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_multipolar_components.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_bend.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_cfd.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_bend.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thick_slice_rbend.h')]

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


class ThickSliceQuadrupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = True

    _xofields = {'_parent': xo.Ref(Quadrupole), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_cfd.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_srotation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_quadrupole.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thick_slice_quadrupole.h')]

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


class ThickSliceSextupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = True

    _xofields = {'_parent': xo.Ref(Sextupole), **_common_xofields}

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_multipole.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thick_slice_sextupole.h')]

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


class ThickSliceOctupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = True

    _xofields = {'_parent': xo.Ref(Octupole), **_common_xofields}

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_multipole.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thick_slice_octupole.h')]

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


class ThickSliceSolenoid(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = True

    _xofields = {'_parent': xo.Ref(Solenoid), **_common_xofields}

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thick_slice_solenoid.h')]

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


class DriftSliceBend(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    allow_loss_refinement = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Bend), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_slice_bend.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['parent_name'] = self.parent_name
        return dct

    copy = _slice_copy

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj.parent_name = dct['parent_name']
        return obj

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out


class DriftSliceRBend(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    allow_loss_refinement = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(RBend), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_slice_rbend.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['parent_name'] = self.parent_name
        return dct

    copy = _slice_copy

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj.parent_name = dct['parent_name']
        return obj

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out


class DriftSliceQuadrupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    allow_loss_refinement = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Quadrupole), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_slice_quadrupole.h')]

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
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out


class DriftSliceSextupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    allow_loss_refinement = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Sextupole), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_slice_sextupole.h')]

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
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out


class DriftSliceOctupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    allow_loss_refinement = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Octupole), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_slice_octupole.h')]

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
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out


class DriftSlice(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    allow_loss_refinement = True
    _force_moveable = True
    isthick = True
    _inherit_strengths = False

    _xofields = {'_parent': xo.Ref(Drift), **_common_xofields}

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_slice.h')]

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
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out