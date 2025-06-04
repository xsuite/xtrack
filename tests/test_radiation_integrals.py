import xtrack as xt
import xobjects as xo

import numpy as np
import pathlib

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

def test_radiation_wiggler():
    env = xt.load_madx_lattice(test_data_folder / 'sps_thick/sps.seq')
    env.vars.load_madx(test_data_folder / 'sps_thick/lhc_q20.str')
    line = env.sps

    line['actcse.31632'].voltage = 4.2e+08
    line['actcse.31632'].frequency = 3e6
    line['actcse.31632'].lag = 180.

    line.particle_ref = xt.Particles(energy0=20e9, mass0=xt.ELECTRON_MASS_EV)
    env.particle_ref = line.particle_ref

    # Wiggler parameters
    k0_wig = 5e-3
    tilt_rad = np.pi/2

    lenwig = 25
    numperiods = 20
    lambdawig = lenwig / numperiods

    wig = Wiggler(period=lambdawig, amplitude=k0_wig, num_periods=numperiods,
                    angle_rad=tilt_rad, scheme='121a')

    tt = line.get_table()
    wig_elems = []
    for name, element in wig.wiggler_dict.items():
        env.elements[name] = element['element']
        wig_elems.append(name)

    wig_line = env.new_line(components=[
                            env.new('s.wig', 'Marker'),
                            wig_elems,
                            env.new('e.wig', 'Marker'),
    ])

    line.insert(wig_line, anchor='start', at=1, from_='qd.31710@end')

    env['sps_thick'] = env.sps.copy(shallow=True)

    line.discard_tracker()
    slicing_strategies = [
        xt.Strategy(slicing=xt.Teapot(1)),  # Default
        xt.Strategy(slicing=xt.Teapot(2), element_type=xt.Bend),
        xt.Strategy(slicing=xt.Teapot(2), element_type=xt.RBend),
        xt.Strategy(slicing=xt.Teapot(8), element_type=xt.Quadrupole),
        xt.Strategy(slicing=xt.Teapot(20), name='mwp.*'),
    ]
    line.slice_thick_elements(slicing_strategies)

    tw4d = line.twiss4d(radiation_integrals=True)
    tw6d = line.twiss()

    line.configure_radiation(model='mean')

    tw_rad = line.twiss(eneloss_and_damping=True, strengths=True)

    print('ex rad int:', tw4d.rad_int_eq_gemitt_x)
    print('ex Chao:   ', tw_rad.eq_gemitt_x)
    print('ey rad int:', tw4d.rad_int_eq_gemitt_y)
    print('ey Chao:   ', tw_rad.eq_gemitt_y)

    print('damping rate x [s^-1] rad int:   ', tw4d.rad_int_damping_constant_x_s)
    print('damping rate x [s^-1] eigenval:  ', tw_rad.damping_constants_s[0])
    print('damping rate y [s^-1] rad int:   ', tw4d.rad_int_damping_constant_y_s)
    print('damping rate y [s^-1] eigenval:  ', tw_rad.damping_constants_s[1])
    print('damping rate z [s^-1] rad int:   ', tw4d.rad_int_damping_constant_zeta_s)
    print('damping rate z [s^-1] eigenval:  ', tw_rad.damping_constants_s[2])

    xo.assert_allclose(
        tw4d.rad_int_eq_gemitt_x, tw_rad.eq_gemitt_x, rtol=1e-3, atol=0)
    xo.assert_allclose(
        tw4d.rad_int_eq_gemitt_y, tw_rad.eq_gemitt_y, rtol=5e-3, atol=0)
    xo.assert_allclose(
        tw4d.rad_int_damping_constant_x_s, tw_rad.damping_constants_s[0],
        rtol=1e-3, atol=0)
    xo.assert_allclose(
        tw4d.rad_int_damping_constant_y_s, tw_rad.damping_constants_s[1],
        rtol=5e-3, atol=0)
    xo.assert_allclose(
        tw4d.rad_int_damping_constant_zeta_s, tw_rad.damping_constants_s[2],
        rtol=1e-3, atol=0)


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