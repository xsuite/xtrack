from shared_knobs import VarSharing

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

    def build_trackers(self, **kwargs):
        for nn, ll in self.lines.items():
            ll.build_tracker(**kwargs)

    def __getitem__(self, key):
        return self.lines[key]

    def __getattr__(self, key):
        return self.lines[key]

    @property
    def vars(self):
        if self._var_sharing is not None:
            return self._var_sharing._vref


