import xtrack as xt
import numpy as np

import xobjects as xo

env = xt.load(['../../test_data/sps_thick/sps.seq',
               '../../test_data/sps_thick/lhc_q20.str'])
line = env.sps

line['actcse.31632'].voltage = 4.2e+08
line['actcse.31632'].frequency = 3e6
line['actcse.31632'].lag = 180.

tt = line.get_table()

# 20 GeV electrons (like in LEP times)
env.particle_ref = xt.Particles(energy0=20e9, mass0=xt.ELECTRON_MASS_EV)
line.particle_ref = env.particle_ref

#####################
# Build the wiggler #
#####################

# Wiggler parameters
env['wig_length'] = 25
env['wig_num_periods'] = 20
env['wig_period_length'] = 'wig_length / wig_num_periods'
env['wig_pole_length'] = '0.25 * wig_period_length'
env['wig_k0'] = 5e-3
env['wig_h0'] = 0
env['wig_tilt_rad'] = np.pi/2

# Assemble wiggler
env.new('wig_pole', 'Bend',
        length='wig_pole_length',
        k0='wig_k0',
        h=0, # Straight magnet (no reference frame curvature)
        k0_from_h=False, # Control k0 and h independently
        rot_s_rad='wig_tilt_rad')

env.new_line(name='wig_period', components=[
    env.new('wig_pole_1', 'wig_pole', k0='-wig_k0', h='-wig_h0'),
    env.new('wig_pole_2', 'wig_pole', k0='wig_k0', h='wig_h0'),
    env.new('wig_pole_3', 'wig_pole', k0='wig_k0', h='wig_h0'),
    env.new('wig_pole_4', 'wig_pole', k0='-wig_k0', h='-wig_h0'),
])

env['wiggler'] = env['wig_num_periods'] * env['wig_period']
env['wiggler'].replace_all_repeated_elements()

env['wiggler_sliced'] = env['wiggler'].copy(shallow=True)
env['wiggler_sliced'].slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(10, mode='thick'))
    ])

tw_wig = env['wiggler_sliced'].twiss(betx=1, bety=1, strengths=True)

line = env['sps']
line.insert(env['wiggler_sliced'], anchor='start', at=1, from_='qd.31710@end')

tt = line.get_table()
tw4d= line.twiss4d(radiation_integrals=True, strengths=True)
tw6d_thick = line.twiss()

line.configure_radiation(model='mean')

tw_rad = line.twiss(eneloss_and_damping=True, strengths=True)

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1, figsize=(6.4, 4.8))
ax1=plt.subplot(2, 1, 1)
tw_wig.plot('x y', ax=ax1)
plt.ylim(-1e-3, 1e-3)
ax2=plt.subplot(2, 1, 2, sharex=ax1)
tw_wig.plot('dx dy', ax=ax2)
plt.subplots_adjust(right=.76, left=0.15)

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
