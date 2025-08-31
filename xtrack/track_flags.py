import numpy as np

flag_mapping = {
    'XS_FLAG_BACKTRACK': 0,
    'XS_FLAG_KILL_CAVITY_KICK': 2,
    'XS_FLAG_IGNORE_GLOBAL_APERTURE': 3,
    'XS_FLAG_IGNORE_LOCAL_APERTURE': 4,
    'XS_FLAG_SR_TAPER': 5,
    'XS_FLAG_SR_KICK_SAME_AS_FIRST': 6
}

flag_defaults = {
    'XS_FLAG_BACKTRACK': False,
    'XS_FLAG_KILL_CAVITY_KICK': False,
    'XS_FLAG_IGNORE_GLOBAL_APERTURE': False,
    'XS_FLAG_IGNORE_LOCAL_APERTURE': False,
    'XS_FLAG_SR_TAPER': False,
    'XS_FLAG_SR_KICK_SAME_AS_FIRST': False
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

    c_header_flag_mapping = c_header_flag_mapping

    def __init__(self, **kwargs):
        object.__setattr__(self, 'flags', flag_defaults.copy())
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_flags_register(self):

        reg = 0
        for flag_name, bit_pos in flag_mapping.items():
            assert bit_pos >= 0
            assert bit_pos < 64
            if self.flags.get(flag_name, False):
                reg |= (1 << bit_pos)
        return np.uint64(reg)

    def __getattr__(self, name):
        if name == 'flags':
            return object.__getattribute__(self, 'flags')
        if name in self.flags:
            return self.flags[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == 'flags': # for unpickling
            object.__setattr__(self, 'flags', value)
            return
        if name not in self.flags:
            raise KeyError(f"Unknown flag '{name}'")
        self.flags[name] = value

    def __repr__(self):
        return f"TrackFlags({self.flags})"

    def print_flag_register(self):
        reg = self.get_flags_register()
        print(f"Flag register (binary): {bin(reg)[2:].zfill(64)}")
