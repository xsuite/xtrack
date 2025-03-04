import xtrack as xt
import numpy as np

env = xt.load_madx_lattice('../../test_data/sps_thick/sps.seq')
env.vars.load_madx('../../test_data/sps_thick/lhc_q20.str')
line = env.sps

line['actcse.31632'].voltage = 4.2e+08
line['actcse.31632'].frequency = 3e6
line['actcse.31632'].lag = 180.

tt = line.get_table()


line.particle_ref = xt.Particles(energy0=20e9, mass0=xt.ELECTRON_MASS_EV)

import wiggler as wgl

k0_wig = 1e-3
lenwig = 10.
numperiods = 50
tilt_rad = np.pi/2


lambdawig = lenwig / numperiods
rhowig = 1 / (k0_wig + 1e-9)

wig = wgl.Wiggler(period=lambdawig, amplitude=k0_wig, num_periods=numperiods,
                  angle_rad=tilt_rad, scheme='121a')

tt = line.get_table()
s_wig = tt['s', 'actcsg.31780']
to_place = []
for name, element in wig.wiggler_dict.items():
    env.elements[name] = element['element']
    to_place = env.place(name, at=s_wig+element['position'])

line.insert(to_place)

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

tw4d = line.twiss4d()
tw6d = line.twiss()

line.configure_radiation(model='mean')

tw_rad = line.twiss(eneloss_and_damping=True, strengths=True)