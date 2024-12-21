from .environment import Environment
from .multiline_legacy import MultilineLegacy
import xtrack as xt

# For backward compatibility
class Multiline(Environment):

    def __init__(self, lines, link_vars=True, **kwargs):

        if not link_vars:
            raise NotImplementedError

        super().__init__(**kwargs)
        for nn, ll in lines.items():
            self.import_line(line=ll, suffix_for_common_elements='__'+nn,
                line_name=nn)


    @classmethod
    def from_dict(cls, dct, **kwargs):
        if 'xsuite_data_type' in dct and dct['xsuite_data_type'] == 'Environment':
            # TODO: Needs to be sorted, returns environment
            return super().from_dict(dct, **kwargs)
        else:
            return xt.MultilineLegacy.from_dict(dct, **kwargs)

    from_madx = MultilineLegacy.from_madx
