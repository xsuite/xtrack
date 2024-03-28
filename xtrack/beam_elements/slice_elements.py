import xobjects as xo
from pathlib import Path
import numpy as np

from ..general import _pkg_root
from ..random import RandomUniform, RandomExponential
from .elements import SynchrotronRadiationRecord, Quadrupole
from ..base_element import BeamElement

xo.context_default.kernels.clear()

class ThinSliceQuadrupole(BeamElement):
    allow_rot_and_shift = False
    _xofields = {
        '_parent': xo.Ref(Quadrupole),
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
        'weight': xo.Float64,
    }

    _skip_in_to_dict = ['_parent']

    _depends_on = [RandomUniform, RandomExponential]

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/synrad_spectrum.h'),
        _pkg_root.joinpath('beam_elements/elements_src/track_multipole.h'),
        _pkg_root.joinpath('beam_elements/elements_src/thin_slice_quadrupole.h'),
    ]

    _internal_record_class = SynchrotronRadiationRecord

    has_backtrack = True

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['_parent_name'] = self._parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj._parent_name = dct['_parent_name']
        return obj