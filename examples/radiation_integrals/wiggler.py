import numpy as np
import xtrack as xt

# NOTE: Put in an option to specify the spacing. Does nothing yet,
# something for later.

class Wiggler:
    def __init__(self, period, amplitude, num_periods, angle_rad=0,
                 scheme='121s'):
        # The scheme_library is a list of all the possible schemes that can be
        # used. The scheme determines the order of the dipoles in the wiggler.
        # The 's' and 'a' stand for a symmetric/antisymmetric configuration
        # respectively.
        self.scheme_library = ['121s', '121a']

        self.wiggler_period = period
        self.wiggler_amplitude = amplitude
        self.wiggler_num_periods = num_periods
        self.angle_rad = angle_rad
        self.scheme = scheme
        self.spacing = 0
        self.wiggler = self._build_wiggler_()
        self.wiggler_dict = self._build_dict_()

    def _build_wiggler_(self):
        wiggler = []

        if self.scheme == '121s':
            for i in range(self.wiggler_num_periods + 1):
                if i != 0 and i != self.wiggler_num_periods:
                    wiggler += [
                        xt.Bend(length=self.wiggler_period / 4,
                                k0=-self.wiggler_amplitude, h=0,
                                rot_s_rad=self.angle_rad),
                        xt.Bend(length=self.wiggler_period / 4,
                                k0=-self.wiggler_amplitude, h=0,
                                rot_s_rad=self.angle_rad),
                        xt.Bend(length=self.wiggler_period / 4,
                                k0=self.wiggler_amplitude, h=0,
                                rot_s_rad=self.angle_rad),
                        xt.Bend(length=self.wiggler_period / 4,
                                k0=self.wiggler_amplitude, h=0,
                                rot_s_rad=self.angle_rad)
                    ]

                elif i == 0:
                    wiggler += [
                        xt.Bend(length=self.wiggler_period / 4,
                                k0=self.wiggler_amplitude, h=0,
                                rot_s_rad=self.angle_rad)
                    ]

                else:
                    wiggler += [
                        xt.Bend(length=self.wiggler_period / 4,
                                k0=-self.wiggler_amplitude, h=0,
                                rot_s_rad=self.angle_rad),
                        xt.Bend(length=self.wiggler_period / 4,
                                k0=-self.wiggler_amplitude, h=0,
                                rot_s_rad=self.angle_rad),
                        xt.Bend(length=self.wiggler_period / 4,
                                k0=self.wiggler_amplitude, h=0,
                                rot_s_rad=self.angle_rad)
                    ]

        if self.scheme == '121a':
            for i in range(self.wiggler_num_periods):
                sign = 1 if i % 2 == 0 else -1
                wiggler += [
                    xt.Bend(length=self.wiggler_period / 4,
                            k0=-sign * self.wiggler_amplitude, h=0,
                            rot_s_rad=self.angle_rad),
                    xt.Bend(length=self.wiggler_period / 4,
                            k0=sign * self.wiggler_amplitude, h=0,
                            rot_s_rad=self.angle_rad),
                    xt.Bend(length=self.wiggler_period / 4,
                            k0=sign * self.wiggler_amplitude, h=0,
                            rot_s_rad=self.angle_rad),
                    xt.Bend(length=self.wiggler_period / 4,
                            k0=-sign * self.wiggler_amplitude, h=0,
                            rot_s_rad=self.angle_rad)
                ]

        print(f'wiggler.shape = {len(wiggler)}')

        return wiggler

    def _get_wiggler_names_(self, wiggler, wiggler_number='1'):
        wiggler_names = []
        for i in range(len(wiggler)):
            wiggler_names += ['mwp' + str(i + 1) + '.' + wiggler_number]

        print(f'wiggler_names.shape = {len(wiggler_names)}')

        return wiggler_names

    def _get_element_positions_(self, wiggler):
        ele_pos = np.zeros(len(wiggler))
        for i in range(1, len(wiggler)):
            ele_pos[i] = ele_pos[i - 1] + wiggler[i - 1].length + self.spacing

        print(f'ele_pos.shape = {ele_pos.shape}')

        return ele_pos

    def _build_dict_(self):
        wiggler = self._build_wiggler_()
        wiggler_names = self._get_wiggler_names_(wiggler)
        ele_pos = self._get_element_positions_(wiggler)
        wiggler_dict = {
            name: {'element': obj, 'position': pos}
            for name, obj, pos in zip(wiggler_names, wiggler, ele_pos)
        }

        return wiggler_dict