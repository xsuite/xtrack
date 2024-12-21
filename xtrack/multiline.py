from .environment import Environment
from .multiline_legacy import MultilineLegacy
import xtrack as xt

# For backward compatibility
class Multiline(Environment):

    @classmethod
    def from_dict(cls, dct, **kwargs):
        if 'xsuite_data_type' in dct and dct['xsuite_data_type'] == 'Environment':
            # TODO: Needs to be sorted, returns environment
            return super().from_dict(dct, **kwargs)
        else:
            return xt.MultilineLegacy.from_dict(dct, **kwargs)
