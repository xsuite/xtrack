import xobjects as xo

from ..general import _pkg_root
from ..base_element import BeamElement
from .elements import (SynchrotronRadiationRecord, Bend, Quadrupole, Sextupole,
                       Octupole)
from ..random import RandomUniform, RandomExponential

_common_xofields = {
    'weight': xo.Float64,
}


_thick_slice_bend_xofields = {
    '_parent': xo.Ref(Bend)}
_thick_slice_bend_xofields.update(_common_xofields)
class ThickSliceBend(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True

    _xofields = _thick_slice_bend_xofields

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_bend.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_cfd.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_bend.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thick_slice_bend.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj


_thick_slice_quadrupole_xofields = {
    '_parent': xo.Ref(Quadrupole)}
_thick_slice_quadrupole_xofields.update(_common_xofields)
class ThickSliceQuadrupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True

    _xofields = _thick_slice_quadrupole_xofields

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_thick_cfd.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_srotation.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_quadrupole.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thick_slice_quadrupole.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj


_thick_slice_sextupole_xofields = {
    '_parent': xo.Ref(Sextupole)}
_thick_slice_sextupole_xofields.update(_common_xofields)
class ThickSliceSextupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True

    _xofields = _thick_slice_sextupole_xofields

    _depends_on = [RandomUniform, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_multipole.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thick_slice_sextupole.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj


_thick_slice_octupole_xofields = {
    '_parent': xo.Ref(Octupole)}
_thick_slice_octupole_xofields.update(_common_xofields)
class ThickSliceOctupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = True
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True

    _xofields = _thick_slice_octupole_xofields

    _depends_on = [RandomUniform, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_multipole.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thick_slice_octupole.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj


_drift_slice_bend_xofields = {
    '_parent': xo.Ref(Bend)}
_drift_slice_bend_xofields.update(_common_xofields)
class DriftSliceBend(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True

    _xofields = _drift_slice_bend_xofields

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_slice_bend.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj

_drift_slice_quadrupole_xofields = {
    '_parent': xo.Ref(Quadrupole)}
_drift_slice_quadrupole_xofields.update(_common_xofields)
class DriftSliceQuadrupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True

    _xofields = _drift_slice_quadrupole_xofields

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_slice_quadrupole.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj


_drift_slice_sextupole_xofields = {
    '_parent': xo.Ref(Sextupole)}
_drift_slice_sextupole_xofields.update(_common_xofields)
class DriftSliceSextupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True

    _xofields = _drift_slice_sextupole_xofields

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_slice_sextupole.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj


_drift_slice_octupole_xofields = {
    '_parent': xo.Ref(Octupole)}
_drift_slice_octupole_xofields.update(_common_xofields)
class DriftSliceOctupole(BeamElement):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True
    isthick = True

    _xofields = _drift_slice_octupole_xofields

    _extra_c_sources = [
        _pkg_root.joinpath('beam_elements/elements_src/drift.h'),
        _pkg_root.joinpath('beam_elements/elements_src/drift_slice_octupole.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj