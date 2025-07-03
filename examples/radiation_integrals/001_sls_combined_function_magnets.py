import xtrack as xt
import numpy as np
import xobjects as xo

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

tw_integrals = line.twiss(radiation_integrals=True)

print('ex rad int:', tw_integrals.rad_int_eq_gemitt_x)
print('ex Chao:   ', tw_rad.eq_gemitt_x)
print('ey rad int:', tw_integrals.rad_int_eq_gemitt_y)
print('ey Chao:   ', tw_rad.eq_gemitt_y)

print('damping rate x [s^-1] rad int:   ', tw_integrals.rad_int_damping_constant_x_s)
print('damping rate x [s^-1] eigenval:  ', tw_rad.damping_constants_s[0])
print('damping rate y [s^-1] rad int:   ', tw_integrals.rad_int_damping_constant_y_s)
print('damping rate y [s^-1] eigenval:  ', tw_rad.damping_constants_s[1])
print('damping rate z [s^-1] rad int:   ', tw_integrals.rad_int_damping_constant_zeta_s)
print('damping rate z [s^-1] eigenval:  ', tw_rad.damping_constants_s[2])

xo.assert_allclose(
    tw_integrals.rad_int_eq_gemitt_x, tw_rad.eq_gemitt_x, rtol=1e-3, atol=0)
xo.assert_allclose(
    tw_integrals.rad_int_eq_gemitt_y, tw_rad.eq_gemitt_y, rtol=5e-3, atol=1e-20)
xo.assert_allclose(
    tw_integrals.rad_int_damping_constant_x_s, tw_rad.damping_constants_s[0],
    rtol=1e-3, atol=0)
xo.assert_allclose(
    tw_integrals.rad_int_damping_constant_y_s, tw_rad.damping_constants_s[1],
    rtol=5e-3, atol=0)
xo.assert_allclose(
    tw_integrals.rad_int_damping_constant_zeta_s, tw_rad.damping_constants_s[2],
    rtol=1e-3, atol=0)