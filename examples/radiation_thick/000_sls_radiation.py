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

tt = line.get_table()
tw4d_thick = line.twiss4d()
tw6d_thick = line.twiss()

line.build_tracker()
line.configure_radiation(model='mean')

tt_bend = tt.rows[tt.element_type == 'Bend']
tt_quad = tt.rows[tt.element_type == 'Quadrupole']
for nn in tt_bend.name:
    line.get(nn).model = 'mat-kick-mat'
    line.get(nn).integrator = 'yoshida4'
    line[nn].num_multipole_kicks = 20
# # for nn in tt_quad.name:
# #     line[nn].radiation_flag = 0

tw_rad_thick = line.twiss(eneloss_and_damping=True, strengths=True)

print('Done thick')

# Sliced thick
env['ring_sliced_thick'] = env.ring.copy(shallow=True)
line_sliced_thick = env['ring_sliced_thick']

line_sliced_thick.discard_tracker()
slicing_strategies = [
    xt.Strategy(slicing=None),  # Default
    xt.Strategy(slicing=xt.Teapot(2, mode='thick'), element_type=xt.Bend),
    xt.Strategy(slicing=xt.Teapot(2, mode='thick'), element_type=xt.Quadrupole),
]
line_sliced_thick.slice_thick_elements(slicing_strategies)

assert line['ars12_mqua_5940..0'].radiation_flag == 10

tw_rad_sliced_thick = line_sliced_thick.twiss(eneloss_and_damping=True, strengths=True)

# Thin ...
env['ring_thin'] = env.ring.copy(shallow=True)
line_thin = env['ring_thin']

line_thin.discard_tracker()
slicing_strategies = [
    xt.Strategy(slicing=None),  # Default
    xt.Strategy(slicing=xt.Teapot(20), element_type=xt.Bend),
    xt.Strategy(slicing=xt.Teapot(8), element_type=xt.Quadrupole),
]
line_thin.slice_thick_elements(slicing_strategies)

line_thin.build_tracker()
# line_thin.configure_radiation(model='mean')

tw_rad_thin = line_thin.twiss(eneloss_and_damping=True, strengths=True)

# Compare tunes, chromaticities, damping rates, equilibrium emittances
print('Tune comparison')
print('Thick:        ', tw_rad_thick.qx, tw_rad_thick.qy)
print('Sliced thick: ', tw_rad_sliced_thick.qx, tw_rad_sliced_thick.qy)
print('Thin:         ', tw_rad_thin.qx, tw_rad_thin.qy)

print('Chromaticity comparison')
print('Thick:        ', tw_rad_thick.dqx, tw_rad_thick.dqy)
print('Sliced thick: ', tw_rad_sliced_thick.dqx, tw_rad_sliced_thick.dqy)
print('Thin:         ', tw_rad_thin.dqx, tw_rad_thin.dqy)

print('Energy loss: ')
print('Thick:        ', tw_rad_thick.eneloss_turn)
print('Sliced thick: ', tw_rad_sliced_thick.eneloss_turn)
print('Thin:         ', tw_rad_thin.eneloss_turn)

print('Partition numbers: ')
print('Thick:        ', tw_rad_thick.partition_numbers)
print('Sliced thick: ', tw_rad_sliced_thick.partition_numbers)
print('Thin:         ', tw_rad_thin.partition_numbers)

print('Equilibrium emittances')
print('Thick:        ', tw_rad_thick.eq_gemitt_x, tw_rad_thick.eq_gemitt_y, tw_rad_thick.eq_gemitt_zeta)
print('Sliced thick: ', tw_rad_sliced_thick.eq_gemitt_x, tw_rad_sliced_thick.eq_gemitt_y, tw_rad_sliced_thick.eq_gemitt_zeta)
print('Thin:         ', tw_rad_thin.eq_gemitt_x, tw_rad_thin.eq_gemitt_y, tw_rad_thin.eq_gemitt_zeta)