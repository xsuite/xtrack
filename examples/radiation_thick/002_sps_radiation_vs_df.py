import xtrack as xt
import numpy as np

env = xt.load_madx_lattice('../../test_data/sps_thick/sps.seq')
env.vars.load_madx('../../test_data/sps_thick/lhc_q20.str')

# RF set tp stay in the linear region
env['actcse.31632'].voltage = 2500e6
env['actcse.31632'].frequency = 3e6
env['actcse.31632'].lag = 180.

line = env.sps
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, energy0=10e9)

line.insert('zeta_shift', obj=xt.ZetaShift(), at=0)

tt = line.get_table()

# line.slice_thick_elements(slicing_strategies=[
#     xt.slicing.Strategy(slicing=None), # Default
#     xt.slicing.Strategy(slicing=xt.Teapot(2, mode='thick'), element_type=xt.RBend),
#     xt.slicing.Strategy(slicing=xt.Teapot(8, mode='thick'), element_type=xt.Quadrupole),
# ])

tw4d = line.twiss4d()
tw6d = line.twiss()

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)

# Prepare trim
env['frev0'] = 1. / tw4d.T_rev0
env['circum'] = tw4d.circumference
env['frev_trim'] = 0.

env['zeta_shift'].dzeta = 'circum * frev_trim / frev0'

dfrev = np.linspace(-1, 0.9, 100)
part_x = []
part_y = []
part_zeta = []
damp_cons_x_s = []
demp_const_y_s = []
damp_const_zeta_s = []
eq_gemitt_x = []
eq_gemitt_y = []
eq_gemitt_zeta = []
rad_int_dconst_x_s =[]
delta_ave = []
for dff in dfrev:
    print(f'dfrev: {dff}')
    env['frev_trim'] = dff
    tw = line.twiss(eneloss_and_damping=True,
                    radiation_integrals=True)
    part_x.append(tw.partition_numbers[0])
    part_y.append(tw.partition_numbers[1])
    part_zeta.append(tw.partition_numbers[2])
    eq_gemitt_x.append(tw.eq_gemitt_x)
    eq_gemitt_y.append(tw.eq_gemitt_y)
    eq_gemitt_zeta.append(tw.eq_gemitt_zeta)
    delta_ave.append(tw.delta.mean())

    damp_cons_x_s.append(tw.damping_constants_s[0])
    damp_const_zeta_s.append(tw.damping_constants_s[2])
    demp_const_y_s.append(tw.damping_constants_s[1])

    rad_int_dconst_x_s.append(tw.rad_int_damping_constant_x_s)

# Cast to numpy arrays
part_x = np.array(part_x)
part_y = np.array(part_y)
part_zeta = np.array(part_zeta)
eq_gemitt_x = np.array(eq_gemitt_x)
eq_gemitt_y = np.array(eq_gemitt_y)
eq_gemitt_zeta = np.array(eq_gemitt_zeta)
delta_ave = np.array(delta_ave)
damp_cons_x_s = np.array(damp_cons_x_s)
demp_const_y_s = np.array(demp_const_y_s)
damp_const_zeta_s = np.array(damp_const_zeta_s)
rad_int_dconst_x_s = np.array(rad_int_dconst_x_s)

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1, figsize=(6.4, 4.8*1.8))
ax1 = plt.subplot(3, 1, 1)
plt.plot(dfrev, delta_ave*1e3)
plt.grid()
plt.ylabel(r'$\delta_\text{ave}$ $[10^{-3}]$')
plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(dfrev, part_x, label='n_x')
plt.plot(dfrev, part_y, label='n_y')
plt.plot(dfrev, part_zeta, label='n_zeta')
plt.plot(dfrev, part_x + part_y + part_zeta, label='total', color='black')
plt.ylabel('Partition numbers')
plt.grid()
plt.legend()
plt.subplot(3, 1, 3, sharex=ax1)
plt.semilogy(dfrev, eq_gemitt_x, label='eq_gemitt_x')
plt.semilogy(dfrev, eq_gemitt_y, label='eq_gemitt_y')
plt.semilogy(dfrev, eq_gemitt_zeta, label='eq_gemitt_zeta')
plt.xlabel(r'$\Delta f_\text{rev}$ [Hz]')
plt.ylim(1e-10, 1e-1)
plt.legend()
plt.grid(True)

plt.show()




