import xobjects as xo

from ..base_element import BeamElement
from ..random import RandomUniformAccurate, RandomExponential
from .elements import SynchrotronRadiationRecord

ID_RADIATION_FROM_PARENT = 10

COMMON_SLICE_XO_FIELDS = {
    'radiation_flag': xo.Field(xo.Int64, default=ID_RADIATION_FROM_PARENT),
    'delta_taper': xo.Float64,
    'weight': xo.Float64,
}

class _SliceBase:

    allow_rot_and_shift = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['parent_name'] = self.parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj.parent_name = dct['parent_name']
        return obj

    def copy(self, **kwargs):
        out = BeamElement.copy(self, **kwargs)
        out._parent = None
        out.parent_name = self.parent_name
        return out