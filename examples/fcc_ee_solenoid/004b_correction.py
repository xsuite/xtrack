import xtrack as xt

from cpymad.madx import Madx

fname = 'fccee_z'; pc_gev = 45.6
fname = 'fccee_t'; pc_gev = 182.5


line = xt.Line.from_json(fname + '_with_sol.json')

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
        xt.TargetSet(['x', 'px', 'y', 'py'], value=tw_sol_off, at='ip.1', tag='orbit'),
        xt.TargetRmatrix(
                    r13=0, r14=0, r23=0, r24=0, # Y-X block
                    r31=0, r32=0, r41=0, r42=0, # X-Y block,
                    start='pqc2le.4', end='ip.1', tol=1e-5, tag='coupl'),
        xt.Target('mux', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('muy', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('betx', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=1, tol=1e-5),
        xt.Target('bety', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=10, tol=1e-6),
        xt.Target('alfx', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),
        xt.Target('alfy', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),

    ]
)


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
        xt.TargetSet(['x', 'px', 'y', 'py'], value=tw_sol_off, at='ip.1', tag='orbit'),
        xt.TargetRmatrix(r13=0, r14=0, r23=0, r24=0, # Y-X block
                         r31=0, r32=0, r41=0, r42=0, # X-Y block,
                         start='ip.1', end='pqc2re.1', tol=1e-5, tag='coupl'),
        xt.Target('mux', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('muy', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('betx', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=1, tol=1e-5),
        xt.Target('bety', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=10, tol=1e-6),
        xt.Target('alfx', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),
        xt.Target('alfy', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),

    ]
)

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

tw_local_corr_back = line.twiss(start='ip.4', end='_end_point', init_at='ip.4',
                                init=tw_local_corr)


line.to_json(fname + '_with_sol_corrected.json')


# plot
import matplotlib.pyplot as plt
plt.close('all')

plt.figure(2)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw_local.s, tw_local.x*1e3, label='x')
plt.plot(tw_local_corr.s, tw_local_corr.x*1e3, label='x corr')
plt.ylabel('x [mm]')

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw_local.s, tw_local.y*1e3, label='y')
plt.plot(tw_local_corr.s, tw_local_corr.y*1e3, label='y corr')

plt.xlabel('s [m]')
plt.ylabel('y [mm]')

plt.figure(3)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw_local.s, tw_local.betx2)
plt.plot(tw_local_corr.s, tw_local_corr.betx2)
plt.plot(tw_local_corr_back.s, tw_local_corr_back.betx2)
plt.ylabel(r'$\beta_{x,2}$ [m]')

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw_local.s, tw_local.bety1)
plt.plot(tw_local_corr.s, tw_local_corr.bety1)
plt.plot(tw_local_corr_back.s, tw_local_corr_back.bety1)
plt.ylabel(r'$\beta_{y,1}$ [m]')

plt.xlabel('s [m]')


tw_sol_on_corrected = line.twiss(method='4d')

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
plt.plot(tw_sol_on.s, tw_sol_on.x_prime, label='correction off')
plt.plot(tw_sol_on.s, tw_sol_on_corrected.x_prime, label='correction on')
plt.ylabel(r'$\beta_{x,2}$ [m]')
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw_sol_on.s, tw_sol_on.y_prime, label='correction off')
plt.plot(tw_sol_on.s, tw_sol_on_corrected.y_prime, label='correction on')
plt.ylabel(r'$\beta_{y,1}$ [m]')

plt.show()