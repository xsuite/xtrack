import xtrack as xt
from scipy.constants import c as clight
from scipy.constants import e as qe

import numpy as np

fname = 'fccee_z'; pc_gev = 45.6
# fname = 'fccee_t'; pc_gev = 182.5


line = xt.Line.from_json(fname + '_with_sol.json')

line.vars['on_sol_ip.1'] = 0
tw_sol_off = line.twiss(method='4d')
line.vars['on_sol_ip.1'] = 1
tw_sol_on = line.twiss(method='4d')
tw_local = line.twiss(start='ip.7', end='ip.2', init_at='ip.1',
                      init=tw_sol_off)

opt_l = line.match(
    solve=False,
    method='4d', n_steps_max=30,
    start='pqc2le.4', end='ip.1', init=tw_sol_off, init_at=xt.START,
    vary=[
        xt.VaryList(['acb1h.l1', 'acb2h.l1','acb1v.l1', 'acb2v.l1'], step=1e-8, tag='corr_l'),
        xt.VaryList(['ks1.l1', 'ks2.l1', 'ks3.l1', 'ks0.l1'], step=1e-7, tag='skew_l'),
        xt.VaryList(['corr_k1.l1', 'corr_k2.l1', 'corr_k3.l1', 'corr_k0.l1'], step=1e-6, tag='normal_l'),
    ],
    targets=[
        xt.TargetSet(['x', 'y'], value=tw_sol_off, tol=1e-7, at='ip.1', tag='orbit'),
        xt.TargetSet(['px', 'py'], value=tw_sol_off, tol=1e-10, at='ip.1', tag='orbit'),
        xt.TargetRmatrix(
                    r13=0, r14=0, r23=0, r24=0, # Y-X block
                    r31=0, r32=0, r41=0, r42=0, # X-Y block,
                    start='pqc2le.4', end='ip.1', tol=1e-6, tag='coupl'),
        xt.Target('mux', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('muy', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('betx', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=1, tol=1e-5),
        xt.Target('bety', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=10, tol=1e-6),
        xt.Target('alfx', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),
        xt.Target('alfy', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),

    ]
)


for iter in range(2):
    # Orbit alone
    opt_l.disable_all_targets(); opt_l.disable_all_vary()
    opt_l.enable_targets(tag='orbit'); opt_l.enable_vary(tag='corr_l'); opt_l.solve()

    # Coupling alone
    opt_l.disable_all_targets(); opt_l.disable_all_vary()
    opt_l.enable_targets(tag='coupl'); opt_l.enable_vary(tag='skew_l'); opt_l.solve()

    # phase, beta and alpha alone
    opt_l.disable_all_targets(); opt_l.disable_all_vary()
    opt_l.enable_vary(tag='normal_l')
    opt_l.enable_targets(tag='mu_ip'); opt_l.solve()
    opt_l.enable_targets(tag='bet_ip'); opt_l.solve()
    opt_l.enable_targets(tag='alf_ip'); opt_l.solve()

# All together
opt_l.enable_all_targets()
opt_l.enable_all_vary()
opt_l.solve()


opt_r = line.match(
    solve=False,
    method='4d', n_steps_max=30,
    start='ip.1', end='pqc2re.1', init=tw_sol_off, init_at=xt.END,
    vary=[
        xt.VaryList(['acb1h.r1', 'acb2h.r1','acb1v.r1', 'acb2v.r1'], step=1e-8, tag='corr_r'),
        xt.VaryList(['ks1.r1', 'ks2.r1', 'ks3.r1', 'ks0.r1'], step=1e-7, tag='skew_r'),
        xt.VaryList(['corr_k1.r1', 'corr_k2.r1', 'corr_k3.r1', 'corr_k0.r1'], step=1e-6, tag='normal_r'),
    ],
    targets=[
        xt.TargetSet(['x', 'y'], value=tw_sol_off, tol=1e-7, at='ip.1', tag='orbit'),
        xt.TargetSet(['px', 'py'], value=tw_sol_off, tol=1e-10, at='ip.1', tag='orbit'),
        xt.TargetRmatrix(r13=0, r14=0, r23=0, r24=0, # Y-X block
                         r31=0, r32=0, r41=0, r42=0, # X-Y block,
                         start='ip.1', end='pqc2re.1', tol=1e-6, tag='coupl'),
        xt.Target('mux', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('muy', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('betx', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=1, tol=1e-5),
        xt.Target('bety', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=10, tol=1e-6),
        xt.Target('alfx', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),
        xt.Target('alfy', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),

    ]
)

for iter in range(2):
    # Orbit alone
    opt_r.disable_all_targets(); opt_r.disable_all_vary()
    opt_r.enable_targets(tag='orbit'); opt_r.enable_vary(tag='corr_r'); opt_r.solve()

    # Coupling alone
    opt_r.disable_all_targets(); opt_r.disable_all_vary()
    opt_r.enable_targets(tag='coupl'); opt_r.enable_vary(tag='skew_r'); opt_r.solve()

    # phase, beta and alpha alone
    opt_r.disable_all_targets(); opt_r.disable_all_vary()
    opt_r.enable_vary(tag='normal_r')
    opt_r.enable_targets(tag='mu_ip'); opt_r.solve()
    opt_r.enable_targets(tag='bet_ip'); opt_r.solve()
    opt_r.enable_targets(tag='alf_ip'); opt_r.solve()

# All together
opt_r.enable_all_targets()
opt_r.enable_all_vary()
opt_r.solve()

tw_local_corr = line.twiss(start='ip.4', end='_end_point', init_at='ip.1',
                            init=tw_sol_off)
line.to_json(fname + '_with_sol_corrected.json')

tw_sol_on_corrected = line.twiss(method='4d')

assert_allclose = np.testing.assert_allclose

# Check that tilt is present
assert_allclose(tw_sol_off['kin_xprime', 'ip.1'], np.tan(0.015), atol=1e-14, rtol=0)

# Check that solenoid introduces coupling
assert tw_sol_on.c_minus > 1e-4

# Check correction
tw_chk = tw_sol_on_corrected

assert_allclose(tw_chk['x', 'ip.1'], 0, atol=1e-8, rtol=0)
assert_allclose(tw_chk['y', 'ip.1'], 0, atol=1e-10, rtol=0)
assert_allclose(tw_chk['kin_xprime', 'ip.1'], tw_sol_off['kin_xprime', 'ip.1'],  atol=1e-9, rtol=0)
assert_allclose(tw_chk['kin_yprime', 'ip.1'], 0,  atol=1e-8, rtol=0)
assert_allclose(tw_chk['x', 'pqc2re.1'], 0, atol=5e-8, rtol=0)
assert_allclose(tw_chk['y', 'pqc2re.1'], 0, atol=5e-8, rtol=0)
assert_allclose(tw_chk['kin_xprime', 'pqc2re.1'], 0, atol=1e-8, rtol=0)
assert_allclose(tw_chk['kin_yprime', 'pqc2re.1'], 0, atol=1e-8, rtol=0)
assert_allclose(tw_chk['x', 'pqc2le.4'], 0, atol=5e-8, rtol=0)
assert_allclose(tw_chk['y', 'pqc2le.4'], 0, atol=5e-8, rtol=0)
assert_allclose(tw_chk['kin_xprime', 'pqc2le.4'], 0, atol=1e-8, rtol=0)
assert_allclose(tw_chk['kin_yprime', 'pqc2le.4'], 0, atol=1e-8, rtol=0)

assert_allclose(tw_chk['betx', 'ip.1'], tw_sol_off['betx', 'ip.1'], atol=0, rtol=5e-4)
assert_allclose(tw_chk['bety', 'ip.1'], tw_sol_off['bety', 'ip.1'], atol=0, rtol=5e-4)
assert_allclose(tw_chk['alfx', 'ip.1'], tw_sol_off['alfx', 'ip.1'], atol=1e-4, rtol=0)
assert_allclose(tw_chk['alfy', 'ip.1'], tw_sol_off['alfy', 'ip.1'], atol=1e-4, rtol=0)
assert_allclose(tw_chk['mux', 'ip.1'], tw_sol_off['mux', 'ip.1'], atol=1e-5, rtol=0)
assert_allclose(tw_chk['muy', 'ip.1'], tw_sol_off['muy', 'ip.1'], atol=1e-5, rtol=0)

assert_allclose(tw_chk['betx', 'pqc2re.1'], tw_sol_off['betx', 'pqc2re.1'], atol=0, rtol=5e-4)
assert_allclose(tw_chk['bety', 'pqc2re.1'], tw_sol_off['bety', 'pqc2re.1'], atol=0, rtol=5e-4)
assert_allclose(tw_chk['alfx', 'pqc2re.1'], tw_sol_off['alfx', 'pqc2re.1'], atol=2e-4, rtol=5e-4)
assert_allclose(tw_chk['alfy', 'pqc2re.1'], tw_sol_off['alfy', 'pqc2re.1'], atol=2e-4, rtol=5e-4)
assert_allclose(tw_chk['mux', 'pqc2re.1'], tw_sol_off['mux', 'pqc2re.1'], atol=1e-5, rtol=5e-5)
assert_allclose(tw_chk['muy', 'pqc2re.1'], tw_sol_off['muy', 'pqc2re.1'], atol=1e-5, rtol=5e-5)

assert_allclose(tw_chk['betx', 'pqc2le.4'], tw_sol_off['betx', 'pqc2le.4'], atol=0, rtol=5e-4)
assert_allclose(tw_chk['bety', 'pqc2le.4'], tw_sol_off['bety', 'pqc2le.4'], atol=0, rtol=5e-4)
assert_allclose(tw_chk['alfx', 'pqc2le.4'], tw_sol_off['alfx', 'pqc2le.4'], atol=2e-4, rtol=5e-4)
assert_allclose(tw_chk['alfy', 'pqc2le.4'], tw_sol_off['alfy', 'pqc2le.4'], atol=2e-4, rtol=5e-4)
assert_allclose(tw_chk['mux', 'pqc2le.4'], tw_sol_off['mux', 'pqc2le.4'], atol=1e-5, rtol=5e-5)
assert_allclose(tw_chk['muy', 'pqc2le.4'], tw_sol_off['muy', 'pqc2le.4'], atol=1e-5, rtol=5e-5)

assert tw_chk.c_minus < 1e-6
assert_allclose(tw_chk['betx2', 'ip.1'] / tw_chk['betx', 'ip.1'], 0, atol=1e-10)
assert_allclose(tw_chk['bety1', 'ip.1'] / tw_chk['bety', 'ip.1'], 0, atol=1e-10)
assert_allclose(tw_chk['betx2', 'pqc2re.1'] / tw_chk['betx', 'pqc2re.1'], 0, atol=1e-10)
assert_allclose(tw_chk['bety1', 'pqc2re.1'] / tw_chk['bety', 'pqc2re.1'], 0, atol=1e-10)
assert_allclose(tw_chk['betx2', 'pqc2le.4'] / tw_chk['betx', 'pqc2le.4'], 0, atol=1e-10)
assert_allclose(tw_chk['bety1', 'pqc2le.4'] / tw_chk['bety', 'pqc2le.4'], 0, atol=1e-10)

tt = line.get_table(attr=True)
s_ip = tt['s', 'ip.1']
P0_J = line.particle_ref.p0c[0] * qe / clight
brho = P0_J / qe / line.particle_ref.q0
Bz = tt.ks * brho

line_rad = line.copy()
line_rad.build_tracker()
line_rad.configure_radiation(model='mean')
line_rad.vars['voltca1'] = line.vv['voltca1_ref']
line_rad.vars['voltca2'] = line.vv['voltca2_ref']
line_rad.compensate_radiation_energy_loss()
tw_sol_on_corr_rad = line_rad.twiss()
mask_len = tt.length > 0
dE = -(np.diff(tw_sol_on_corr_rad.ptau) * tw_sol_on_corr_rad.particle_on_co.energy0[0])
dE_ds = tt.s * 0
dE_ds[mask_len] = dE[mask_len[:-1]] / tt.length[mask_len]

# plot
import matplotlib.pyplot as plt
plt.close('all')

plt.figure(5)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw_sol_on.s, tw_sol_on.betx2, label='correction off')
plt.plot(tw_sol_on.s, tw_sol_on_corrected.betx2, label='correction on')
plt.ylabel(r'$\beta_{x,2}$ [m]')
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw_sol_on.s, tw_sol_on.bety1, label='correction off')
plt.plot(tw_sol_on.s, tw_sol_on_corrected.bety1, label='correction on')
plt.ylabel(r'$\beta_{y,1}$ [m]')

plt.figure(6)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw_sol_on.s, tw_sol_off.kin_xprime, label='solenoid off')
plt.plot(tw_sol_on.s, tw_sol_on.kin_xprime, label='correction off')
plt.plot(tw_sol_on.s, tw_sol_on_corrected.kin_xprime, label='correction on')
plt.ylabel("x'")
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw_sol_on.s, tw_sol_off.kin_yprime, label='solenoid off')
plt.plot(tw_sol_on.s, tw_sol_on.kin_yprime, label='correction off')
plt.plot(tw_sol_on.s, tw_sol_on_corrected.kin_yprime, label='correction on')
plt.ylabel("y'")

plt.figure(7)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw_sol_on.s, tw_sol_on.x - tw_sol_off.x, label='correction off')
plt.plot(tw_sol_on.s, tw_sol_on_corrected.x  - tw_sol_off.x, label='correction on')
plt.ylabel("x")
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw_sol_on.s, tw_sol_on.y - tw_sol_off.y, label='correction off')
plt.plot(tw_sol_on.s, tw_sol_on_corrected.y  - tw_sol_off.y, label='correction on')
plt.ylabel("y")
plt.suptitle('Solenoid tilt is subtracted')

plt.figure(8)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw_sol_on.s, tw_sol_on.kin_xprime - tw_sol_off.kin_xprime, label='correction off')
plt.plot(tw_sol_on.s, tw_sol_on_corrected.kin_xprime  - tw_sol_off.kin_xprime, label='correction on')
plt.ylabel("x'")
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw_sol_on.s, tw_sol_on.kin_yprime - tw_sol_off.kin_yprime, label='correction off')
plt.plot(tw_sol_on.s, tw_sol_on_corrected.kin_yprime  - tw_sol_off.kin_yprime, label='correction on')
plt.ylabel("y'")
plt.suptitle('Solenoid tilt is subtracted')

plt.figure(70, figsize=(6.4, 4.8*1.5))

ax1 = plt.subplot(3, 1, 1)
mask_ip = tt.name == 'ip.1'
plt.plot(tt.s[~mask_ip] - s_ip, Bz[~mask_ip])
plt.ylabel('Bz [T]')
plt.grid()

ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(tw_sol_on.s - s_ip, tw_sol_on_corrected.x  - tw_sol_off.x, label='correction on')
plt.ylabel("x [m]")
plt.grid()

ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(tw_sol_on.s - s_ip, tw_sol_on_corrected.y  - tw_sol_off.y, label='correction on')
plt.ylabel("y [m]")
plt.suptitle('Closed orbit - Solenoid tilt is subtracted')
plt.subplots_adjust(hspace=0.3, top=0.9)
plt.xlim(-5, 5)
plt.xlabel('s [m]')
plt.grid()

for nn in tt.rows['mcb.*'].name:
    for ax in [ax2, ax3]:
        ax.axvline(tt['s', nn] - s_ip, color='k', linestyle='--', alpha=0.3)

plt.figure(80, figsize=(6.4, 4.8*1.5))
s_ip = tt['s', 'ip.1']

ax1 = plt.subplot(3, 1, 1, sharex=ax1)
mask_ip = tt.name == 'ip.1'
plt.plot(tt.s[~mask_ip] - s_ip, Bz[~mask_ip])
plt.ylabel('Bz [T]')
plt.grid()

ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(tw_sol_on.s - s_ip, tw_sol_on_corrected.kin_xprime  - tw_sol_off.kin_xprime)
plt.ylabel("x'")
plt.grid()

ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(tw_sol_on.s - s_ip, tw_sol_on_corrected.kin_yprime  - tw_sol_off.kin_yprime)
plt.ylabel("y'")
plt.suptitle('Angles - Solenoid tilt is subtracted')
plt.subplots_adjust(hspace=0.3, top=0.9)
plt.xlim(-5, 5)
plt.xlabel('s [m]')
plt.grid()

for nn in tt.rows['mcb.*'].name:
    for ax in [ax2, ax3]:
        ax.axvline(tt['s', nn] - s_ip, color='k', linestyle='--', alpha=0.3)

plt.figure(90, figsize=(6.4, 4.8))
ax1 = plt.subplot(2, 1, 1, sharex=ax1)
plt.plot(tt.s[~mask_ip] - s_ip, Bz[~mask_ip])
plt.ylabel('Bz [T]')
plt.grid()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw_sol_on.s - s_ip, dE_ds * 1e-2 * 1e-3, '-', label='dE/ds')
plt.ylabel('dE/ds [keV/cm]')
plt.xlim(-5, 5)
plt.xlabel('s [m]')
plt.grid()
plt.suptitle('Radiated power')

plt.figure(85, figsize=(6.4, 4.8*1.5))
s_ip = tt['s', 'ip.1']

ax1 = plt.subplot(3, 1, 1)
mask_ip = tt.name == 'ip.1'
plt.plot(tt.s[~mask_ip] - s_ip, Bz[~mask_ip])
plt.ylabel('Bz [T]')
plt.grid()

ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(tw_sol_on.s - s_ip, tw_sol_on_corrected.bety1)
plt.ylabel(r"$\beta_{y,1}$")
plt.grid()

ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(tw_sol_on.s - s_ip, tw_sol_on_corrected.betx2)
plt.ylabel(r"$\beta_{x,2}$")
plt.suptitle('Local coupling')
plt.subplots_adjust(hspace=0.3, top=0.9)
plt.xlim(-10, 10)
plt.xlabel('s [m]')
plt.grid()
plt.suptitle('Local coupling')


plt.show()