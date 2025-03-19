import xtrack as xt
import numpy as np

from scipy.constants import hbar
from scipy.constants import electron_volt
from scipy.constants import c as clight

env = xt.load_madx_lattice('b075_2024.09.25.madx')
line = env.ring
line.particle_ref = xt.Particles(energy0=2.7e9, mass0=xt.ELECTRON_MASS_EV)
line.configure_bend_model(num_multipole_kicks=20)

line['vrf'] = 1.8e6
line['frf'] = 499.6e6
line['lagrf'] = 180.

line.insert(
    env.new('cav', 'Cavity', voltage='vrf', frequency='frf', lag='lagrf', at=0))

line_thick = line.copy(shallow=True)

tt = line.get_table()
tw4d_thick = line.twiss4d()
tw6d_thick = line.twiss()

env['ring_thick'] = env.ring.copy(shallow=True)

line.discard_tracker()
slicing_strategies = [
    xt.Strategy(slicing=None),  # Default
    xt.Strategy(slicing=xt.Teapot(20), element_type=xt.Bend),
    xt.Strategy(slicing=xt.Teapot(8), element_type=xt.Quadrupole),
]
line.slice_thick_elements(slicing_strategies)

tw4d = line.twiss4d()
tw6d = line.twiss()

line.configure_radiation(model='mean')
tw_rad = line.twiss(eneloss_and_damping=True, strengths=True)

line_thick.build_tracker()
line_thick.configure_radiation(model='mean')

tt_bend = tt.rows[tt.element_type == 'Bend']
tt_quad = tt.rows[tt.element_type == 'Quadrupole']
for nn in tt_bend.name:
    line.get(nn).model = 'bend-kick-bend'
    line.get(nn).integrator = 'uniform'
    line[nn].num_multipole_kicks = 5
# # for nn in tt_quad.name:
# #     line[nn].radiation_flag = 0

tw_rad_thick = line_thick.twiss(eneloss_and_damping=True, strengths=True)