from .shared_knobs import VarSharing
import xtrack as xt

class Multiline:

    def __init__(self, lines: dict, link_vars=True):
        self.lines = {}
        self.lines.update(lines)

        line_names = list(self.lines.keys())
        line_list = [self.lines[nn] for nn in line_names]
        if link_vars:
            self._var_sharing = VarSharing(lines=line_list, names=line_names)
        else:
            self._var_sharing = None

        for ll in line_list:
            ll._in_multiline = True

    def to_dict(self, include_var_management=True):
        dct = {}
        if include_var_management:
            dct['_var_manager'] = self._var_sharing.manager.dump()
            dct['_var_management_data'] = self._var_sharing.data
        dct['lines'] = {}
        for nn, ll in self.lines.items():
            dct['lines'][nn] = ll.to_dict(include_var_management=False)
        return dct

    @classmethod
    def from_dict(cls, dct):
        lines = {}
        for nn, ll in dct['lines'].items():
            lines[nn] = xt.Line.from_dict(ll)

        new_multiline = cls(lines=lines, link_vars=('_var_manager' in dct))

        if '_var_manager' in dct:
            for kk in dct['_var_management_data'].keys():
                new_multiline._var_sharing.data[kk].update(
                                                dct['_var_management_data'][kk])
            new_multiline._var_sharing.manager.load(dct['_var_manager'])

        return new_multiline

    def build_trackers(self, **kwargs):
        for nn, ll in self.lines.items():
            ll.build_tracker(**kwargs)

    def __getitem__(self, key):
        return self.lines[key]

    def __dir__(self):
        return list(self.lines.keys()) + object.__dir__(self)

    def __getattr__(self, key):
        if key in self.lines:
            return self.lines[key]
        else:
            raise AttributeError(f"Multiline object has no attribute {key}.")

    @property
    def vars(self):
        if self._var_sharing is not None:
            return self._var_sharing._vref

