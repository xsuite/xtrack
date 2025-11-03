import xtrack as xt
import xdeps as xd
import numpy as np
import xobjects as xo
import xpart as xp
from scipy.constants import e as qe
from scipy.constants import c as clight
from scipy.constants import hbar

# env = xt.load('../../test_data/sps_thick/sps.seq')
# env.vars.load('../../test_data/sps_thick/lhc_q20.str')
# line = env.sps
# line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=20e9,
#                                  )
# line['qf.62410'].shift_y = 1e-3

line = xt.load('lep_corrected_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]

line.ref['on_sol'] = 1.
line['sol_l_ip2'].ks *= line.ref['on_sol']
line['sol_r_ip2'].ks *= line.ref['on_sol']
line['sol_l_ip4'].ks *= line.ref['on_sol']
line['sol_r_ip4'].ks *= line.ref['on_sol']
line['sol_l_ip6'].ks *= line.ref['on_sol']
line['sol_r_ip6'].ks *= line.ref['on_sol']
line['sol_l_ip8'].ks *= line.ref['on_sol']
line['sol_r_ip8'].ks *= line.ref['on_sol']

# No skew quads
for nn in line.vars.get_table().rows['kqt.*'].name:
    line[nn] = 0

line['on_sol'] = 1.

# I flip the bumps (Jorg had probably opposite solenoids)
line['on_spin_bumps'] = 1.
line['kcv32.l2'] = ' 3.14467e-05 * on_spin_bumps * (-1)'
line['kcv26.l2'] = ' 6.28933e-05 * on_spin_bumps * (-1)'
line['kcv20.l2'] = ' 3.14467e-05 * on_spin_bumps * (-1)'
line['kcv20.r2'] = '-3.14467e-05 * on_spin_bumps * (-1)'
line['kcv26.r2'] = '-6.28933e-05 * on_spin_bumps * (-1)'
line['kcv32.r2'] = '-3.14467e-05 * on_spin_bumps * (-1)'
line['kcv32.l4'] = ' 5.21432e-05 * on_spin_bumps * (-1)'
line['kcv26.l4'] = ' 0.000104286 * on_spin_bumps * (-1)'
line['kcv20.l4'] = ' 5.21432e-05 * on_spin_bumps * (-1)'
line['kcv20.r4'] = '-5.21432e-05 * on_spin_bumps * (-1)'
line['kcv26.r4'] = '-0.000104286 * on_spin_bumps * (-1)'
line['kcv32.r4'] = '-5.21432e-05 * on_spin_bumps * (-1)'
line['kcv32.l6'] = '  1.3513e-05 * on_spin_bumps * (-1)'
line['kcv26.l6'] = '  2.7026e-05 * on_spin_bumps * (-1)'
line['kcv20.l6'] = '  1.3513e-05 * on_spin_bumps * (-1)'
line['kcv20.r6'] = ' -1.3513e-05 * on_spin_bumps * (-1)'
line['kcv26.r6'] = ' -2.7026e-05 * on_spin_bumps * (-1)'
line['kcv32.r6'] = ' -1.3513e-05 * on_spin_bumps * (-1)'
line['kcv32.l8'] = ' 4.67179e-05 * on_spin_bumps * (-1)'
line['kcv26.l8'] = ' 9.34358e-05 * on_spin_bumps * (-1)'
line['kcv20.l8'] = ' 4.67179e-05 * on_spin_bumps * (-1)'
line['kcv20.r8'] = '-4.67179e-05 * on_spin_bumps * (-1)'
line['kcv26.r8'] = '-9.34358e-05 * on_spin_bumps * (-1)'
line['kcv32.r8'] = '-4.67179e-05 * on_spin_bumps * (-1)'

line['on_spin_bumps'] = 1; line['on_sol'] = 0
tw_off = line.twiss4d(spin=True, radiation_integrals=True)
line['on_spin_bumps'] = 1; line['on_sol'] = 1
tw = line.twiss4d(spin=True, radiation_integrals=True)
tw_ir4 = tw.rows[9997:11200:'s']

for ttww in [tw_off, tw]:

    kappa_x = ttww.rad_int_kappa_x
    kappa_y = ttww.rad_int_kappa_y
    kappa = ttww.rad_int_kappa
    iv_x = ttww.rad_int_iv_x
    iv_y = ttww.rad_int_iv_y
    iv_z = ttww.rad_int_iv_z

    n0_iv = ttww.spin_x * iv_x + ttww.spin_y * iv_y + ttww.spin_z * iv_z
    r0 = ttww.particle_on_co.get_classical_particle_radius0()
    m0_J = ttww.particle_on_co.mass0 * qe
    m0_kg = m0_J / clight**2

    # reference https://lib-extopc.kek.jp/preprints/PDF/1980/8011/8011060.pdf

    alpha_plus_co = 1. / ttww.circumference * np.sum(
        kappa**3 * (1 - 2./9. * n0_iv**2) * ttww.length)

    tp_inv = 5 * np.sqrt(3) / 8 * r0 * hbar * ttww.gamma0**5 / m0_kg * alpha_plus_co
    tp_s = 1 / tp_inv
    tp_turn = tp_s / ttww.T_rev0

    brho_ref = ttww.particle_on_co.p0c[0] / clight / ttww.particle_on_co.q0
    brho_part = (brho_ref * ttww.particle_on_co.rvv[0] * ttww.particle_on_co.energy[0]
                / ttww.particle_on_co.energy0[0])

    By = kappa_x * brho_part
    Bx = -kappa_y * brho_part
    Bz = ttww.ks * brho_ref
    B_mod = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_mod[B_mod == 0] = 999. # avoid division by zero

    ib_x = Bx / B_mod
    ib_y = By / B_mod
    ib_z = Bz / B_mod

    n0_ib = ttww.spin_x * ib_x + ttww.spin_y * ib_y + ttww.spin_z * ib_z

    alpha_minus_co = 1. / ttww.circumference * np.sum(kappa**3 * n0_ib *  ttww.length)

    pol_inf = 8 / 5 / np.sqrt(3) * alpha_minus_co / alpha_plus_co

    ttww._data['alpha_plus_co'] = alpha_plus_co
    ttww._data['alpha_minus_co'] = alpha_minus_co
    ttww._data['pol_inf'] = pol_inf
    ttww['n0_ib'] = n0_ib

# Check normalization and closure of spin vector
xo.assert_allclose(tw.spin_x**2 + tw.spin_y**2 + tw.spin_z**2,
                   1, atol=1e-12, rtol=0)
xo.assert_allclose(tw.spin_x[0], tw.spin_x[-1], atol=1e-10, rtol=0)
xo.assert_allclose(tw.spin_y[0], tw.spin_y[-1], atol=1e-10, rtol=0)
xo.assert_allclose(tw.spin_z[0], tw.spin_z[-1], atol=1e-10, rtol=0)

line['vrfc231'] = 15

line.configure_radiation(model='mean')
line['on_spin_bumps'] = 0.; line['on_sol'] = 0.
tw_rad_off = line.twiss(spin=True, eneloss_and_damping=True)
line['on_spin_bumps'] = 0.; line['on_sol'] = 1.
tw_rad_on = line.twiss(spin=True, eneloss_and_damping=True)

line['on_spin_bumps'] = 0.; line['on_sol'] = 0.
p_off = xp.generate_matched_gaussian_bunch(
    line=line,
    nemitt_x=tw_rad_off.eq_nemitt_x,
    nemitt_y=tw_rad_off.eq_nemitt_y,
    sigma_z=np.sqrt(tw_rad_off.eq_gemitt_zeta * tw_rad_off.bets0),
    num_particles=100,
    engine='linear')
# Need to patch the longitudinal plane
p_off.zeta += tw_rad_off.zeta[0]
p_off.delta += tw_rad_off.delta[0]

p_off.spin_x = tw_rad_off.spin_x[0]
p_off.spin_y = tw_rad_off.spin_y[0]
p_off.spin_z = tw_rad_off.spin_z[0]

line['on_spin_bumps'] = 0.; line['on_sol'] = 1.
p_on = xp.generate_matched_gaussian_bunch(
    line=line,
    nemitt_x=tw_rad_on.eq_nemitt_x,
    nemitt_y=tw_rad_on.eq_nemitt_y,
    sigma_z=np.sqrt(tw_rad_on.eq_gemitt_zeta * tw_rad_on.bets0),
    num_particles=100,
    engine='linear')
# Need to patch the longitudinal plane
p_on.zeta += tw_rad_on.zeta[0]
p_on.delta += tw_rad_on.delta[0]

p_on.spin_x = tw_rad_on.spin_x[0]
p_on.spin_y = tw_rad_on.spin_y[0]
p_on.spin_z = tw_rad_on.spin_z[0]

line.configure_radiation(model='quantum')

line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=10))

line['on_spin_bumps'] = 0.; line['on_sol'] = 0.
line.track(p_off, num_turns=1000, turn_by_turn_monitor=True,
           with_progress=10)
mon_off = line.record_last_track

pol_x_off = mon_off.spin_x.mean(axis=0)
pol_y_off = mon_off.spin_y.mean(axis=0)
pol_z_off = mon_off.spin_z.mean(axis=0)
pol_off = np.sqrt(pol_x_off**2 + pol_y_off**2 + pol_z_off**2)

line['on_spin_bumps'] = 0.; line['on_sol'] = 1.
line.track(p_on, num_turns=1000, turn_by_turn_monitor=True,
              with_progress=10)
mon_on = line.record_last_track

pol_x_on = mon_on.spin_x.mean(axis=0)
pol_y_on = mon_on.spin_y.mean(axis=0)
pol_z_on = mon_on.spin_z.mean(axis=0)
pol_on = np.sqrt(pol_x_on**2 + pol_y_on**2 + pol_z_on**2)

import matplotlib.pyplot as plt
plt.close('all')
fig2 = plt.figure(2)
ax1 = fig2.add_subplot(211)
tw.plot(lattice_only=True, ax=ax1)
plt.plot(tw.s, tw.spin_x, '-', label='spin_x')
plt.plot(tw.s, tw.spin_z, '-', label='spin_y')
plt.xlabel('s [m]')
plt.ylabel('spin component')
ax2 = fig2.add_subplot(212, sharex=ax1)
tw.plot('x y', ax=ax2)
plt.xlabel('s [m]')
plt.ylabel('x, y [m]')

plt.figure(3)
plt.plot(-tw_ir4.spin_z, -tw_ir4.spin_x, 'x-', label='spin')
plt.xlabel(r'$n_{0z}$')
plt.ylabel(r'$n_{0x}$')
plt.axis('equal')
plt.suptitle('IP4 right')

plt.figure(4)
plt.plot(pol_off, label='spin pol off')
plt.plot(pol_on, label='spin pol on')
plt.xlabel('turns')
plt.ylabel('Polarization')
plt.legend()

plt.show()