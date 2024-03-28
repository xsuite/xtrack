import xobjects as xo
from pathlib import Path
import numpy as np

from ..general import _pkg_root
from ..random import RandomUniform, RandomExponential
from .elements import SynchrotronRadiationRecord, Quadrupole, Sextupole, Bend
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

_thin_slice_quad_xofields = {
    '_parent': xo.Ref(Quadrupole)}
_thin_slice_quad_xofields.update(_common_xofields)
class ThinSliceQuadrupole(BeamElement):
    allow_rot_and_shift = False
    _skip_in_to_dict = ['_parent']
    _depends_on = [RandomUniform, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord
    has_backtrack = True

    _xofields = _thin_slice_quad_xofields

    _extra_c_sources = _common_c_sources + [
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_quadrupole.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj

_thin_slice_sext_xofields = {
    '_parent': xo.Ref(Sextupole)}
_thin_slice_sext_xofields.update(_common_xofields)
class ThinSliceSextupole(BeamElement):
    allow_rot_and_shift = False
    _skip_in_to_dict = ['_parent']
    _depends_on = [RandomUniform, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord
    has_backtrack = True

    _xofields = _thin_slice_sext_xofields

    _extra_c_sources = _common_c_sources + [
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_sextupole.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj

_thin_slice_bend_xofields = {
    '_parent': xo.Ref(Bend)}
_thin_slice_bend_xofields.update(_common_xofields)
class ThinSliceBend(BeamElement):
    allow_rot_and_shift = False
    _skip_in_to_dict = ['_parent']
    _depends_on = [RandomUniform, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord
    has_backtrack = True

    _xofields = _thin_slice_bend_xofields

    _extra_c_sources = _common_c_sources + [
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_bend.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj

_thin_slice_bend_entry_xofields = {
    '_parent': xo.Ref(Bend)}
_thin_slice_bend_entry_xofields.update(_common_xofields)
class ThinSliceBendEntry(BeamElement):
    allow_rot_and_shift = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True

    _xofields = _thin_slice_bend_entry_xofields

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_bend_entry.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj

_thin_slice_bend_exit_xofields = {
    '_parent': xo.Ref(Bend)}
_thin_slice_bend_exit_xofields.update(_common_xofields)
class ThinSliceBendExit(BeamElement):
    allow_rot_and_shift = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True

    _xofields = _thin_slice_bend_exit_xofields

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_bend_exit.h')]

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj