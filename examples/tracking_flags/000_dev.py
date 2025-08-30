import xtrack as xt
import numpy as np

# Todo: track flags in save/reload
# preserve config needs to be preserve_config_and_track_flags

flag_mapping = {
    'XS_FLAG_BACKTRACK': 0,
    'XS_FLAG_TAPER': 1,
    'XS_KILL_CAVITY_KICK': 2
}

flag_defaults = {
    'XS_FLAG_BACKTRACK': False,
    'XS_FLAG_TAPER': False,
    'XS_KILL_CAVITY_KICK': False
}

c_header_flag_mapping = """
#ifndef XSTUITE_TRACK_FLAGS_H
#define XSTUITE_TRACK_FLAGS_H

"""
for flag_name, bit_pos in flag_mapping.items():
    c_header_flag_mapping += f"#define {flag_name} ({bit_pos})\n"
c_header_flag_mapping += """
#endif // XSTUITE_TRACK_FLAGS_H
"""


class TrackFlags:

    def __init__(self, **kwargs):
        object.__setattr__(self, 'flags', flag_defaults.copy())
        for key, value in kwargs.items():
            setattr(self, key, value)

    def make_flags_register(self):

        reg = 0
        for flag_name, bit_pos in flag_mapping.items():
            assert bit_pos >= 0
            assert bit_pos < 64
            if self.flags.get(flag_name, False):
                reg |= (1 << bit_pos)
        return np.uint64(reg)

    def __getattr__(self, name):
        if name in self.flags:
            return self.flags[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        assert value in (True, False)
        if name not in self.flags:
            raise KeyError(f"Unknown flag '{name}'")
        self.flags[name] = value

    def __repr__(self):
        return f"TrackFlags({self.flags})"

    def print_flag_register(self):
        reg = self.make_flags_register()
        print(f"Flag register (binary): {bin(reg)[2:].zfill(64)}")

track_flags = TrackFlags()
track_flags.print_flag_register()

track_flags.XS_FLAG_BACKTRACK = True
track_flags.print_flag_register()

track_flags.XS_KILL_CAVITY_KICK = True
track_flags.print_flag_register()