import xtrack as xt
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=2e9)

n_bends_per_cell = 6
n_cells_par_arc = 3
n_arcs = 3

n_bends = n_bends_per_cell * n_cells_par_arc * n_arcs

env.vars({
    'l.mq': 0.5,
    'kqf': 0.027,
    'kqd': -0.0271,
    'l.mb': 10,
    'l.ms': 0.3,
    'k2sf': 0.001,
    'k2sd': -0.001,
    'angle.mb': 2 * np.pi / n_bends,
    'k0.mb': 'angle.mb / l.mb',
    'k0l.corrector': 0,
    'k1sl.corrector': 0,
    'l.halfcell': 38,
})

halfcell = env.new_line(components=[

    env.new_element('mid', xt.Marker, at='l.halfcell'),

    env.new_element('mb.2', xt.Bend, k0='k0.mb', h='k0.mb', length='l.mb', at='l.halfcell / 2'),
    env.new_element('mb.1', xt.Replica, parent_name='mb.2', at='-l.mb - 1', from_='mb.2'),
    env.new_element('mb.3', xt.Replica, parent_name='mb.2', at='l.mb + 1', from_='mb.2'),

    env.new_element('mq.d', xt.Quadrupole, k1='kqd', length='l.mq', at = '0.5 + l.mq / 2'),
    env.new_element('mq.f', xt.Quadrupole, k1='kqf', length='l.mq', at = 'l.halfcell - l.mq / 2 - 0.5'),

    env.new_element('corrector.v', xt.Multipole, at=1.5),
    env.new_element('corrector.h', xt.Multipole, at='l.halfcell - 1.5'),

    env.new_element('ms.d', xt.Sextupole, length='l.ms', k2='k2sf', at=2.),
    env.new_element('ms.f', xt.Sextupole, length='l.ms', k2='k2sd', at='l.halfcell - 2.'),

])

hcell_left = halfcell.replicate(name='l', mirror=True)
hcell_right = halfcell.replicate(name='r')

cell = env.new_line(components=[
    env.new_element('start', xt.Marker),
    hcell_left,
    hcell_right,
    env.new_element('end', xt.Marker),
])

opt = cell.match(
    method='4d',
    vary=xt.VaryList(['kqf', 'kqd'], step=1e-5),
    targets=xt.TargetSet(
        qx=0.333333,
        qy=0.333333,
    ))
tw_cell = cell.twiss4d()


env.vars({
    'kqf.ss': 0.027 / 2,
    'kqd.ss': -0.0271 / 2,
})

halfcell_ss = env.new_line(components=[

    env.new_element('mid', xt.Marker, at='l.halfcell'),

    env.new_element('mq.ss.d', xt.Quadrupole, k1='kqd.ss', length='l.mq', at = '0.5 + l.mq / 2'),
    env.new_element('mq.ss.f', xt.Quadrupole, k1='kqf.ss', length='l.mq', at = 'l.halfcell - l.mq / 2 - 0.5'),

    env.new_element('corrector.ss.v', xt.Multipole, at=1.5),
    env.new_element('corrector.ss.h', xt.Multipole, at='l.halfcell - 1.5'),
])

hcell_left_ss = halfcell_ss.replicate(name='l', mirror=True)
hcell_right_ss = halfcell_ss.replicate(name='r')
cell_ss = env.new_line(components=[
    env.new_element('start.ss', xt.Marker),
    hcell_left_ss,
    hcell_right_ss,
    env.new_element('end.ss', xt.Marker),
])

opt = cell_ss.match(
    method='4d',
    vary=xt.VaryList(['kqf.ss', 'kqd.ss'], step=1e-5),
    targets=xt.TargetSet(
        betx=tw_cell.betx[-1], bety=tw_cell.bety[-1], at='start.ss',
    ))



arc = env.new_line(components=[
    cell.replicate(name='cell.1'),
    cell.replicate(name='cell.2'),
    cell.replicate(name='cell.3'),
])


ss = env.new_line(components=[
    cell_ss.replicate('cell.1'),
    cell_ss.replicate('cell.2'),
])

ring = env.new_line(components=[
    arc.replicate(name='arc.1'),
    ss.replicate(name='ss.1'),
    arc.replicate(name='arc.2'),
    ss.replicate(name='ss.2'),
    arc.replicate(name='arc.3'),
    ss.replicate(name='ss.3'),
])

## Insretion

env.vars({
    'k1.q1': 0.025,
    'k1.q2': -0.025,
    'k1.q3': 0.025,
    'k1.q4': -0.02,
    'k1.q5': 0.025,
})

half_insertion = env.new_line(components=[
    env.new_element('ip', xt.Marker),
    env.new_element('dd.0', xt.Drift, length=20),
    env.new_element('mq.1', xt.Quadrupole, k1='k1.q1', length='l.mq'),
    env.new_element('dd.1', xt.Drift, length=5),
    env.new_element('mq.2', xt.Quadrupole, k1='k1.q2', length='l.mq'),
    env.new_element('dd.2', xt.Drift, length=12),
    env.new_element('mq.3', xt.Quadrupole, k1='k1.q3', length='l.mq'),
    env.new_element('dd.3', xt.Drift, length=18),
    env.new_element('mq.4', xt.Quadrupole, k1='k1.q4', length='l.mq'),
    env.new_element('dd.4', xt.Drift, length=18),
    env.new_element('mq.5', xt.Quadrupole, k1='k1.q5', length='l.mq'),
    env.new_element('dd.5', xt.Drift, length=0.5),
    env.new_element('e.insertion', xt.Marker),
])

tw_arc = arc.twiss4d()

opt = half_insertion.match(
    solve=False,
    betx=tw_arc.betx[0], bety=tw_arc.bety[0],
    alfx=tw_arc.alfx[0], alfy=tw_arc.alfy[0],
    init_at='e.insertion',
    start='ip', end='e.insertion',
    vary=xt.VaryList(['k1.q1', 'k1.q2', 'k1.q3', 'k1.q4'], step=1e-5),
    targets=[
        xt.TargetSet(alfx=0, alfy=0, at='ip'),
        xt.Target(lambda tw: tw.betx[0] - tw.bety[0], 0),
        xt.Target(lambda tw: tw.betx.max(), xt.LessThan(400)),
        xt.Target(lambda tw: tw.bety.max(), xt.LessThan(400)),
        xt.Target(lambda tw: tw.betx.min(), xt.GreaterThan(2)),
        xt.Target(lambda tw: tw.bety.min(), xt.GreaterThan(2)),
    ]
)
opt.step(40)
opt.solve()

insertion = env.new_line([
    half_insertion.replicate('l', mirror=True),
    half_insertion.replicate('r')])



ring2 = env.new_line(components=[
    arc.replicate(name='arcc.1'),
    ss.replicate(name='sss.2'),
    arc.replicate(name='arcc.2'),
    insertion,
    arc.replicate(name='arcc.3'),
    ss.replicate(name='sss.3')
])

ring2_sliced = ring2.select()
# ring2_sliced.cut_at_s(np.arange(0, ring2.get_length(), 0.5))





import matplotlib.pyplot as plt
plt.close('all')
for ii, rr in enumerate([ring, ring2]):

    tw = rr.twiss4d()

    fig = plt.figure(ii, figsize=(6.4*1.2, 4.8))
    ax1 = fig.add_subplot(2, 1, 1)
    pltbet = tw.plot('betx bety', ax=ax1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    pltdx = tw.plot('dx', ax=ax2)
    fig.subplots_adjust(right=.85)
    pltbet.move_legend(1.2,1)
    pltdx.move_legend(1.2,1)

ring2.survey().plot()


plt.show()



