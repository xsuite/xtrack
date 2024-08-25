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


half_cell.vars['kqf'] = 0.027/2
half_cell.vars['kqd'] = -0.0271/2
half_cell.element_refs['qf1'].k1 = half_cell.vars['kqf']
half_cell.element_refs['qd1'].k1 = half_cell.vars['kqd']

opt_halfcell = half_cell.match(
    method='4d',
    start='start_cell', end='mid_cell',
    init='periodic_symmetric',
    targets=xt.TargetSet(mux=0.2501/2, muy=0.2502/2, at='mid_cell'),
    vary=xt.VaryList(['kqf', 'kqd'], step=1e-5),
)

tw_half_cell = half_cell.twiss4d(strengths=True, init='periodic_symmetric')

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
cell.vars['kqf'] = 0.027/2
cell.vars['kqd'] = -0.0271/2
cell.element_refs['qf1'].k1 = cell.vars['kqf']
cell.element_refs['qd1'].k1 = cell.vars['kqd']

opt_cell = cell.match(
    method='4d',
    start='start_cell', end='end_cell',
    init='periodic',
    targets=xt.TargetSet(mux=0.2501, muy=0.2502, at='end_cell'),
    vary=xt.VaryList(['kqf', 'kqd'], step=1e-5),
)
tw_cell = cell.twiss4d(strengths=True)

xo.assert_allclose(tw_half_cell.betx[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].betx, atol=0, rtol=1e-8)
xo.assert_allclose(tw_half_cell.bety[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].bety, atol=0, rtol=1e-8)
xo.assert_allclose(tw_half_cell.alfx[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].alfx, atol=1e-8, rtol=0)
xo.assert_allclose(tw_half_cell.alfy[:-1], # remove '_end_point'
                   tw_cell.rows[:'mid_cell'].alfy, atol=1e-8, rtol=0)

import matplotlib.pyplot as plt
plt.close('all')

twplt1 = tw_cell.plot()
twplt1.ylim(left_lo=0, right_lo=0.5, right_hi=4)
twplt1.left.figure.suptitle('Full cell (periodic twiss)')
twplt2 = tw_half_cell.plot()
twplt2.ylim(left_lo=0, right_lo=0.5, right_hi=4)
twplt2.left.set_xlim(0, 100)
twplt2.left.figure.suptitle('Half cell (periodic-symmetric twiss)')
plt.show()
