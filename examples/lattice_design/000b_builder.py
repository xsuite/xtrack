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

env.new('mb', xt.Bend, length='l.mb', k0='k0.mb', h='k0.mb')
env.new('mq', xt.Quadrupole, length='l.mq')
env.new('ms', xt.Sextupole, length='l.ms')
env.new('corrector', xt.Multipole, knl=[0], ksl=[0])

env.new('mq.f', 'mq', k1='kqf')
env.new('mq.d', 'mq', k1='kqd')

halfcell = env.new_builder()

# End of the half cell (will be mid of the cell)
halfcell.new('mid', xt.Marker, at='l.halfcell')

# Bends
halfcell.new('mb.2', 'mb', at='l.halfcell / 2')
halfcell.new('mb.1', 'mb', at='-l.mb - 1', from_='mb.2')
halfcell.new('mb.3', 'mb', at='l.mb + 1', from_='mb.2')

# Quads
halfcell.place('mq.d', at = '0.5 + l.mq / 2')
halfcell.place('mq.f', at = 'l.halfcell - l.mq / 2 - 0.5')

# Sextupoles
halfcell.new('ms.d', 'ms', k2='k2sf', at=1.2, from_='mq.d')
halfcell.new('ms.f', 'ms', k2='k2sd', at=-1.2, from_='mq.f')

# Dipole correctors
halfcell.new('corrector.v', 'corrector', at=0.75, from_='mq.d')
halfcell.new('corrector.h', 'corrector', at=-0.75, from_='mq.f')

halfcell = halfcell.build() # The builder can be found in halfcell.builder

hcell_left = halfcell.replicate(suffix='l', mirror=True)
hcell_right = halfcell.replicate(suffix='r')

cell = env.new_builder()

cell.new('start', xt.Marker)
cell.place(hcell_left)
cell.place(hcell_right)
cell.new('end', xt.Marker)

cell = cell.build()

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

halfcell_ss = env.new_builder()

halfcell_ss.new('mid.ss', xt.Marker, at='l.halfcell')

halfcell_ss.new('mq.ss.d', 'mq', k1='kqd.ss', at = '0.5 + l.mq / 2'),
halfcell_ss.new('mq.ss.f', 'mq', k1='kqf.ss', at = 'l.halfcell - l.mq / 2 - 0.5')

halfcell_ss.new('corrector.ss.v', 'corrector', at=0.75, from_='mq.ss.d')
halfcell_ss.new('corrector.ss.h', 'corrector', at=-0.75, from_='mq.ss.f')

halfcell_ss = halfcell_ss.build()

hcell_left_ss = halfcell_ss.replicate(suffix='l', mirror=True)
hcell_right_ss = halfcell_ss.replicate(suffix='r')

cell_ss = env.new_builder()
cell_ss.new('start.ss', xt.Marker)
cell_ss.place(hcell_left_ss)
cell_ss.place(hcell_right_ss)
cell_ss.new('end.ss', xt.Marker)

cell_ss = cell_ss.build()

opt = cell_ss.match(
    method='4d',
    vary=xt.VaryList(['kqf.ss', 'kqd.ss'], step=1e-5),
    targets=xt.TargetSet(
        betx=tw_cell.betx[-1], bety=tw_cell.bety[-1], at='start.ss',
    ))

arc = env.new_builder()
arc.new('cell.1', cell, mode='replica')
arc.new('cell.2', cell, mode='replica')
arc.new('cell.3', cell, mode='replica')
arc = arc.build()

ss = env.new_builder()
ss.new('ss.cell.1', cell_ss, mode='replica')
ss.new('ss.cell.2', cell_ss, mode='replica')
ss = ss.build()

breakpoint()
ring = env.new_builder()
ring.new('arc.1', arc, mode='replica')
ring.new('ss.1', ss, mode='replica')
ring.new('arc.2', arc, mode='replica')
ring.new('ss.2', ss, mode='replica')
ring.new('arc.3', arc, mode='replica')
ring.new('ss.3', ss, mode='replica')
ring = ring.build()

## Insertion

env.vars({
    'k1.q1': 0.025,
    'k1.q2': -0.025,
    'k1.q3': 0.025,
    'k1.q4': -0.02,
    'k1.q5': 0.025,
})

half_insertion = env.new_builder()

half_insertion.new('ip', xt.Marker)
half_insertion.new('e.insertion', xt.Marker, at=76)

half_insertion.new('mq.1', xt.Quadrupole, k1='k1.q1', length='l.mq', at = 20)
half_insertion.new('mq.2', xt.Quadrupole, k1='k1.q2', length='l.mq', at = 25)
half_insertion.new('mq.3', xt.Quadrupole, k1='k1.q3', length='l.mq', at=37)
half_insertion.new('mq.4', xt.Quadrupole, k1='k1.q4', length='l.mq', at=55)
half_insertion.new('mq.5', xt.Quadrupole, k1='k1.q5', length='l.mq', at=73)

half_insertion.new('corrector.ss.1', 'corrector', at=0.75, from_='mq.1')
half_insertion.new('corrector.ss.2', 'corrector', at=-0.75, from_='mq.2')
half_insertion.new('corrector.ss.3', 'corrector', at=0.75, from_='mq.3')
half_insertion.new('corrector.ss.4', 'corrector', at=-0.75, from_='mq.4')
half_insertion.new('corrector.ss.5', 'corrector', at=0.75, from_='mq.5')

half_insertion = half_insertion.build()

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

insertion = env.new_builder()
insertion.new('l', half_insertion, mirror=True)
insertion.new('r', half_insertion)
insertion = insertion.build()

ring2 = env.new_builder()

ring2.place('arc.1')
ring2.place('ss.1')
ring2.place('arc.2')
ring2.place(insertion)
ring2.place('arc.3')
ring2.place('ss.3')

ring2 = ring2.build()

# # Check buffer behavior
ring2_sliced = ring2.select()
ring2_sliced.cut_at_s(np.arange(0, ring2.get_length(), 0.5))


import matplotlib.pyplot as plt
plt.close('all')
for ii, rr in enumerate([ring, ring2_sliced]):

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



