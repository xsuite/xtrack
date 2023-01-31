from shared_knobs import VarSharing


class Multiline:

    def __init__(self, lines: dict):
        self.lines = {}
        self.lines.update(lines)

        line_names = list(self.lines.keys())
        line_list = [self.lines[nn] for nn in line_names]
        self._var_sharing = VarSharing(lines=line_list, names=line_names)

        for ll in line_list:
            ll._in_multiline = True

    def build_trackers(self, **kwargs):
        for nn, ll in self.lines.items():
            ll.build_tracker(**kwargs)

    def __getitem__(self, key):
        return self.lines[key]

    def __getattr__(self, key):
        return self.lines[key]

