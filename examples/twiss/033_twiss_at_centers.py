import xtrack as xt

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

halfcell = env.new_line(components=[

    # End of the half cell (will be mid of the cell)
    env.new('mid', xt.Marker, at='l.halfcell'),

    # Bends
    env.new('mb.2', 'mb', at='l.halfcell / 2'),
    env.new('mb.1', 'mb', at='-l.mb - 1', from_='mb.2'),
    env.new('mb.3', 'mb', at='l.mb + 1', from_='mb.2'),

    # Quads
    env.place('mq.d', at = '0.5 + l.mq / 2'),
    env.place('mq.f', at = 'l.halfcell - l.mq / 2 - 0.5'),

    # Sextupoles
    env.new('ms.d', 'ms', k2='k2sf', at=1.2, from_='mq.d'),
    env.new('ms.f', 'ms', k2='k2sd', at=-1.2, from_='mq.f'),

    # Dipole correctors
    env.new('corrector.v', 'corrector', at=0.75, from_='mq.d'),
    env.new('corrector.h', 'corrector', at=-0.75, from_='mq.f')

])


cell = -halfcell + halfcell

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

    env.new('mid.ss', xt.Marker, at='l.halfcell'),

    env.new('mq.ss.d', 'mq', k1='kqd.ss', at = '0.5 + l.mq / 2'),
    env.new('mq.ss.f', 'mq', k1='kqf.ss', at = 'l.halfcell - l.mq / 2 - 0.5'),

    env.new('corrector.ss.v', 'corrector', at=0.75, from_='mq.ss.d'),
    env.new('corrector.ss.h', 'corrector', at=-0.75, from_='mq.ss.f')
])

cell_ss = env.new_line([
    env.new('start.cell.ss', 'Marker'),
    -halfcell_ss + halfcell_ss
])

opt = cell_ss.match(
    method='4d',
    vary=xt.VaryList(['kqf.ss', 'kqd.ss'], step=1e-5),
    targets=xt.TargetSet(
        betx=tw_cell.betx[-1], bety=tw_cell.bety[-1], at='start.cell.ss'))


arc = 3 * cell
ss = 2 * cell_ss

ring = 3 * (arc + ss)

tw = ring.twiss4d()

line = ring
tt = line.get_table()

s_tol = 1e-8

mask_thick = ~np.isclose(tt.s_center, tt.s_start, atol=s_tol, rtol=0)
tt_thick = tt.rows[mask_thick]
ELEMENT_TYPES_TWISS_CENTER = 'Bend', 'Quadrupole', 'Sextupole', 'Octupole'
mask_match_type = np.zeros(len(tt_thick), dtype=bool)
for element_type in ELEMENT_TYPES_TWISS_CENTER:
    mask_match_type |= tt_thick.element_type == element_type
tt_cut = tt_thick.rows[mask_match_type]

line_twiss = line.copy(shallow=True)

# Create all the markers
insertions = []
for ii, nn in enumerate(tt_cut.name):
    env_nn = tt_cut.env_name[ii]
    nn_center = env_nn + '@center'
    if nn_center in line_twiss.env.elements:
        assert isinstance(line_twiss.env.elements[nn_center], xt.Marker)
    else:
        line_twiss.env.new(nn_center, xt.Marker)

    insertions.append(line.env.place(nn_center, at=tt_cut['s_center', ii]))

line_twiss.insert(insertions)