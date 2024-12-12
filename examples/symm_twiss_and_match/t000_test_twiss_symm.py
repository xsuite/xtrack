import xtrack as xt
import xobjects as xo
import numpy as np

# Build line with half a cell
half_cell = xt.Line(
    elements={
        'start_cell': xt.Marker(),
        'drift0': xt.Drift(length=1.),
        'qf1': xt.Quadrupole(k1=0.027/2, length=1.),
        'drift1_1': xt.Drift(length=1),
        'bend1': xt.Bend(k0=3e-4, h=3e-4, length=45.),
        'drift1_2': xt.Drift(length=1.),
        'qd1': xt.Quadrupole(k1=-0.0271/2, length=1.),
        'drift2': xt.Drift(length=1),
        'mid_cell': xt.Marker(),
    }
)
half_cell.particle_ref = xt.Particles(p0c=2e9)

# Add observation points every 1 m (to see betas inside bends)
half_cell.discard_tracker()
s_cut = np.arange(0, half_cell.get_length(), 1.)
half_cell.cut_at_s(s_cut)

tw_half_cell = half_cell.twiss4d(init='periodic_symmetric', # <--- periodic-symmetric boundary
                                 strengths=True # to get the strengths in table
                                )

cell = xt.Line(
    elements={
        'start_cell': xt.Marker(),
        'drift0': xt.Drift(length=1.),
        'qf1':    xt.Quadrupole(k1=0.027/2, length=1.),
        'drift1': xt.Drift(length=1),
        'bend1':  xt.Bend(k0=3e-4, h=3e-4, length=45.),
        'drift2': xt.Drift(length=1.),
        'qd1':    xt.Quadrupole(k1=-0.0271/2, length=1.),
        'drift3': xt.Drift(length=1),
        'mid_cell': xt.Marker(),
        'drift4': xt.Replica('drift3'),
        'qd2':    xt.Replica('qd1'),
        'drift5': xt.Replica('drift2'),
        'bend2':  xt.Replica('bend1'),
        'drift6': xt.Replica('drift1'),
        'qf2':    xt.Replica('qf1'),
        'drift7': xt.Replica('drift0'),
        'end_cell': xt.Marker(),
    }
)
cell.particle_ref = xt.Particles(p0c=2e9)

# Add observation points every 1 m (to see betas inside bends)
cell.discard_tracker()
s_cut = np.arange(0, cell.get_length(), 1.)
cell.cut_at_s(s_cut)

tw_cell = cell.twiss4d(strengths=True)

xo.assert_allclose(tw_half_cell.betx[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].betx, atol=0, rtol=1e-8)
xo.assert_allclose(tw_half_cell.bety[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].bety, atol=0, rtol=1e-8)
xo.assert_allclose(tw_half_cell.alfx[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].alfx, atol=1e-8, rtol=0)
xo.assert_allclose(tw_half_cell.alfy[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].alfy, atol=1e-8, rtol=0)
xo.assert_allclose(tw_half_cell.dx[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].dx, atol=1e-8, rtol=0)
xo.assert_allclose(tw_half_cell.dpx[:-1], # remove '_end_point'
                     tw_cell.rows[:'mid_cell'].dpx, atol=1e-8, rtol=0)

xo.assert_allclose(tw_half_cell.ax_chrom[:-1], # remove '_end_point'
                    tw_cell.rows[:'mid_cell'].ax_chrom, atol=1e-5, rtol=0)
xo.assert_allclose(tw_half_cell.ay_chrom[:-1], # remove '_end_point'
                    tw_cell.rows[:'mid_cell'].ay_chrom, atol=1e-5, rtol=0)
xo.assert_allclose(tw_half_cell.bx_chrom[:-1], # remove '_end_point'
                    tw_cell.rows[:'mid_cell'].bx_chrom, atol=1e-5, rtol=0)
xo.assert_allclose(tw_half_cell.by_chrom[:-1], # remove '_end_point'
                    tw_cell.rows[:'mid_cell'].by_chrom, atol=1e-5, rtol=0)

xo.assert_allclose(tw_half_cell.qx, tw_cell.qx / 2, atol=1e-9, rtol=0)
xo.assert_allclose(tw_half_cell.qy, tw_cell.qy / 2, atol=1e-9, rtol=0)
xo.assert_allclose(tw_half_cell.dqx, tw_cell.dqx / 2, atol=1e-6, rtol=0)
xo.assert_allclose(tw_half_cell.dqy, tw_cell.dqy / 2, atol=1e-6, rtol=0)

for ll in [cell, half_cell]:
    ll.vars['kqf'] = 0.027/2
    ll.vars['kqd'] = -0.0271/2
    ll.element_refs['qf1'].k1 = ll.vars['kqf']
    ll.element_refs['qd1'].k1 = ll.vars['kqd']

tw_cell = cell.twiss4d(strengths=True)
tw_half_cell = half_cell.twiss4d(init='periodic_symmetric', strengths=True)

xo.assert_allclose(tw_cell.mux[-1], 0.231, atol=1e-3, rtol=0)
xo.assert_allclose(tw_cell.muy[-1], 0.233, atol=1e-3, rtol=0)
xo.assert_allclose(tw_half_cell.mux[-1], 0.231 / 2, atol=1e-3, rtol=0)
xo.assert_allclose(tw_half_cell.muy[-1], 0.233 / 2, atol=1e-3, rtol=0)

opt_halfcell = half_cell.match(
    method='4d',
    start='start_cell', end='mid_cell',
    init='periodic_symmetric',
    targets=xt.TargetSet(mux=0.2501/2, muy=0.2502/2, at='mid_cell'),
    vary=xt.VaryList(['kqf', 'kqd'], step=1e-5),
)

tw_cell = cell.twiss4d(strengths=True)
tw_half_cell = half_cell.twiss4d(init='periodic_symmetric', strengths=True)

xo.assert_allclose(tw_half_cell.mux[-1], 0.2501 / 2, atol=1e-3, rtol=0)
xo.assert_allclose(tw_half_cell.muy[-1], 0.2502 / 2, atol=1e-3, rtol=0)
xo.assert_allclose(tw_cell.mux[-1], 0.231, atol=1e-3, rtol=0) # unaffected
xo.assert_allclose(tw_cell.muy[-1], 0.233, atol=1e-3, rtol=0) # unaffected

opt_cell = cell.match(
    method='4d',
    start='start_cell', end='end_cell',
    init='periodic',
    targets=xt.TargetSet(mux=0.2501, muy=0.2502, at='end_cell'),
    vary=xt.VaryList(['kqf', 'kqd'], step=1e-5),
)

xo.assert_allclose(half_cell.vv['kqf'], cell.vv['kqf'], rtol=1e-9, atol=0)
xo.assert_allclose(half_cell.vv['kqd'], cell.vv['kqd'], rtol=1e-9, atol=0)

tw_cell = cell.twiss4d(strengths=True)
tw_half_cell = half_cell.twiss4d(init='periodic_symmetric', strengths=True)

xo.assert_allclose(tw_half_cell.mux[-1], 0.2501 / 2, atol=1e-3, rtol=0)
xo.assert_allclose(tw_half_cell.muy[-1], 0.2502 / 2, atol=1e-3, rtol=0)
xo.assert_allclose(tw_cell.mux[-1], 0.2501, atol=1e-3, rtol=0)
xo.assert_allclose(tw_cell.muy[-1], 0.2502, atol=1e-3, rtol=0)

xo.assert_allclose(tw_half_cell.betx[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].betx, atol=0, rtol=1e-8)
xo.assert_allclose(tw_half_cell.bety[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].bety, atol=0, rtol=1e-8)
xo.assert_allclose(tw_half_cell.alfx[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].alfx, atol=1e-8, rtol=0)
xo.assert_allclose(tw_half_cell.alfy[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].alfy, atol=1e-8, rtol=0)
xo.assert_allclose(tw_half_cell.dx[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].dx, atol=1e-8, rtol=0)
xo.assert_allclose(tw_half_cell.dpx[:-1], # remove '_end_point'
                     tw_cell.rows[:'mid_cell'].dpx, atol=1e-8, rtol=0)
xo.assert_allclose(tw_half_cell.ddx[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].ddx, atol=1e-7, rtol=0)
xo.assert_allclose(tw_half_cell.ddpx[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].ddpx, atol=1e-8, rtol=0)

xo.assert_allclose(tw_half_cell.ax_chrom[:-1], # remove '_end_point'
                    tw_cell.rows[:'mid_cell'].ax_chrom, atol=1e-5, rtol=0)
xo.assert_allclose(tw_half_cell.ay_chrom[:-1], # remove '_end_point'
                    tw_cell.rows[:'mid_cell'].ay_chrom, atol=1e-5, rtol=0)
xo.assert_allclose(tw_half_cell.bx_chrom[:-1], # remove '_end_point'
                    tw_cell.rows[:'mid_cell'].bx_chrom, atol=1e-5, rtol=0)
xo.assert_allclose(tw_half_cell.by_chrom[:-1], # remove '_end_point'
                    tw_cell.rows[:'mid_cell'].by_chrom, atol=1e-5, rtol=0)

xo.assert_allclose(tw_half_cell.qx, tw_cell.qx / 2, atol=1e-9, rtol=0)
xo.assert_allclose(tw_half_cell.qy, tw_cell.qy / 2, atol=1e-9, rtol=0)
xo.assert_allclose(tw_half_cell.dqx, tw_cell.dqx / 2, atol=1e-6, rtol=0)
xo.assert_allclose(tw_half_cell.dqy, tw_cell.dqy / 2, atol=1e-6, rtol=0)

tw_off_mom_cell = cell.twiss4d(strengths=True, delta0=1e-3)
tw_off_mom_half_cell = half_cell.twiss4d(
    init='periodic_symmetric', strengths=True, delta0=1e-3)

xo.assert_allclose(tw_off_mom_half_cell.x[:-1],
                   tw_off_mom_cell.rows[:'mid_cell'].x, atol=1e-12, rtol=0)

import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4*1.2, 4.8))
ax1 = fig1.add_subplot(211)
plt_cell = tw_cell.plot(ax=ax1)
plt_cell.move_legend(left=1.4, bottom=1)
ax1.set_title('Full cell')

ax2 = fig1.add_subplot(212, sharex=ax1, sharey=ax1)
plt_half_cell = tw_half_cell.plot(ax=ax2)
plt_half_cell.move_legend(left=1.4, bottom=1)
ax2.set_title('Half cell')

ax1.set_xlim(0, tw_cell.s[-1])
fig1.subplots_adjust(right=0.73, hspace=0.5)

fig2 = plt.figure(2, figsize=(6.4*1.2, 4.8))
ax1 = fig2.add_subplot(211)
plt_cell = tw_cell.plot('ddx', ax=ax1)
plt_cell.move_legend(left=1.4, bottom=1)
plt_cell.ax.set_title('Full cell')

ax2 = fig2.add_subplot(212, sharex=ax1, sharey=ax1)
plt_half_cell = tw_half_cell.plot('ddx', ax=ax2)
plt_half_cell.move_legend(left=1.4, bottom=1)
plt_half_cell.ax.set_title('Half cell')

ax1.set_xlim(0, tw_cell.s[-1])
fig2.subplots_adjust(right=0.73, hspace=0.5)



plt.show()
