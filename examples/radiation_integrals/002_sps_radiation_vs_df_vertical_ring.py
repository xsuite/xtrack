import xtrack as xt
import xobjects as xo
import numpy as np

tilt = True

env = xt.load('../../test_data/sps_thick/sps.seq')
env.vars.load('../../test_data/sps_thick/lhc_q20.str')
line = env.sps

line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, energy0=10e9)

line.insert('zeta_shift', obj=xt.ZetaShift(), at=0)

# RF set tp stay in the linear region
env['actcse.31632'].voltage = 2500e6
env['actcse.31632'].frequency = 3e6
env['actcse.31632'].lag = 180.


if tilt:

    tt = line.get_table()
    tt_mb = tt.rows['mb.*']
    tt_lsf = tt.rows['lsf.*']
    tt_lsd = tt.rows['lsd.*']

    for nn in tt_mb.name:
        line[nn].rot_s_rad = np.deg2rad(90)

    # I need skew sextupoles to correct the chromaticity when the dispersion is vertical
    for nn in list(tt_lsf.name) + list(tt_lsd.name):
        line[nn].rot_s_rad = np.deg2rad(-30)

    opt_q = line.match(
        solve=False,
        vary=xt.VaryList(['kqf', 'kqd'], step=1e-4),
        targets=xt.TargetSet(qx=20.18, qy=20.13, tol=1e-4))
    opt_q.solve()

    opt_chrom = line.match(
        solve=False,
        vary=xt.VaryList(['klsfb', 'klsfa', 'klsdb', 'klsda'], step=1e-4),
        targets=xt.TargetSet(dqx=1., dqy=1, tol=1e-4))
    opt_chrom.solve()


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

dfrev = np.linspace(-0.7, 0.7, 21)
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
rad_int_dconst_y_s = []
rad_int_dconst_zeta_s = []
rad_int_ex = []
rad_int_ey = []
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
    rad_int_dconst_y_s.append(tw.rad_int_damping_constant_y_s)
    rad_int_dconst_zeta_s.append(tw.rad_int_damping_constant_zeta_s)

    rad_int_ex.append(tw.rad_int_eq_gemitt_x)
    rad_int_ey.append(tw.rad_int_eq_gemitt_y)

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
rad_int_dconst_y_s = np.array(rad_int_dconst_y_s)
rad_int_dconst_zeta_s = np.array(rad_int_dconst_zeta_s)
rad_int_ex = np.array(rad_int_ex)
rad_int_ey = np.array(rad_int_ey)

xo.assert_allclose(
    rad_int_dconst_x_s, damp_cons_x_s, rtol=0.03, atol=0.05)
xo.assert_allclose(
    rad_int_dconst_y_s, demp_const_y_s, rtol=0.03, atol=0.05)
xo.assert_allclose(
    rad_int_dconst_zeta_s, damp_const_zeta_s, rtol=0.03, atol=0.05)

mask = np.abs(rad_int_dconst_x_s) > 0.25
xo.assert_allclose(
    rad_int_ex[mask], eq_gemitt_x[mask], rtol=0.065, atol=1e-14)
mask = np.abs(rad_int_dconst_y_s) > 0.25
xo.assert_allclose(
    rad_int_ey[mask], eq_gemitt_y[mask], rtol=0.065, atol=1e-14)

if tilt:
    xo.assert_allclose(
        rad_int_ex, 0, rtol=0, atol=1e-14)
else:
    xo.assert_allclose(
        rad_int_ey, 0, rtol=1e-14, atol=1e-14)


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

plt.figure(2, figsize=(6.4, 4.8*1.8))
axd = plt.subplot(3, 1, 1, sharex=ax1)
plt.plot(dfrev, damp_cons_x_s, label='Chao')
plt.plot(dfrev, rad_int_dconst_x_s, label='Rad. Int.')
plt.ylabel(r'$d_x$ [s$^{-1}$]')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2, sharex=ax1, sharey=axd)
plt.plot(dfrev, demp_const_y_s)
plt.plot(dfrev, rad_int_dconst_y_s)
plt.ylabel(r'$d_y$ [s$^{-1}$]')
plt.grid(True)

plt.subplot(3, 1, 3, sharex=ax1, sharey=axd)
plt.plot(dfrev, damp_const_zeta_s)
plt.plot(dfrev, rad_int_dconst_zeta_s)
plt.ylabel(r'$d_\zeta$ [s$^{-1}$]')
plt.xlabel(r'$\Delta f_\text{rev}$ [Hz]')
plt.grid(True)

plt.figure(3, figsize=(6.4, 4.8*1.8))

axemi = plt.subplot(2, 1, 1, sharex=ax1)
plt.plot(dfrev, eq_gemitt_x, label='Chao')
plt.plot(dfrev, rad_int_ex, label='Rad. Int.')
plt.ylabel(r'$\epsilon_x$ [m]')
plt.ylim(-5e-7, 5e-7)
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2, sharex=ax1, sharey=axemi)
plt.plot(dfrev, eq_gemitt_y, label='Chao')
plt.plot(dfrev, rad_int_ey, label='Rad. Int.')
plt.ylabel(r'$\epsilon_y$ [m]')
plt.xlabel(r'$\Delta f_\text{rev}$ [Hz]')
plt.legend()
plt.grid(True)

plt.show()




