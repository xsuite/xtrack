import xtrack as xt
import numpy as np

from scipy.constants import hbar
from scipy.constants import electron_volt
from scipy.constants import c as clight

env = xt.load_madx_lattice('../../test_data/sps_thick/sps.seq')
env.vars.load_madx('../../test_data/sps_thick/lhc_q20.str')
line = env.sps

line['actcse.31632'].voltage = 4.2e+08
line['actcse.31632'].frequency = 3e6
line['actcse.31632'].lag = 180.

tt = line.get_table()


line.particle_ref = xt.Particles(energy0=20e9, mass0=xt.ELECTRON_MASS_EV)
env.particle_ref = line.particle_ref

import wiggler as wgl

# Wiggler parameters
k0_wig = 5e-3
tilt_rad = np.pi/2

lenwig = 25
numperiods = 20
lambdawig = lenwig / numperiods

wig = wgl.Wiggler(period=lambdawig, amplitude=k0_wig, num_periods=numperiods,
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

tt = line.get_table()
tw4d_thick = line.twiss4d()
tw6d_thick = line.twiss()

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