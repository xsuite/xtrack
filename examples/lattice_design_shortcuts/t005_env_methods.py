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

    env.new('mid', xt.Marker, at='l.halfcell'),

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

# Twiss the ring
tw = ring.twiss4d()

# Twiss a single cell
tw_a_cell = ring.twiss4d(start='mq.f::10', end='mq.f::11', init='periodic')

## Insertion

env.vars({
    'k1.q1': 0.025,
    'k1.q2': -0.025,
    'k1.q3': 0.025,
    'k1.q4': -0.02,
    'k1.q5': 0.025,
})

half_insertion = env.new_line(components=[

    # Start-end markers
    env.new('ip', xt.Marker),
    env.new('e.insertion', xt.Marker, at=76),

    # Quads
    env.new('mq.1', xt.Quadrupole, k1='k1.q1', length='l.mq', at = 20),
    env.new('mq.2', xt.Quadrupole, k1='k1.q2', length='l.mq', at = 25),
    env.new('mq.3', xt.Quadrupole, k1='k1.q3', length='l.mq', at=37),
    env.new('mq.4', xt.Quadrupole, k1='k1.q4', length='l.mq', at=55),
    env.new('mq.5', xt.Quadrupole, k1='k1.q5', length='l.mq', at=73),

    # Dipole correctors (will use h and v on the same corrector)
    env.new('corrector.ss.1', 'corrector', at=0.75, from_='mq.1'),
    env.new('corrector.ss.2', 'corrector', at=-0.75, from_='mq.2'),
    env.new('corrector.ss.3', 'corrector', at=0.75, from_='mq.3'),
    env.new('corrector.ss.4', 'corrector', at=-0.75, from_='mq.4'),
    env.new('corrector.ss.5', 'corrector', at=0.75, from_='mq.5'),

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

insertion = -half_insertion + half_insertion

ring2 = 2 * (arc + ss) + arc + insertion

########################

## Line
ring.info("k0.mb") # Get info
ring.info('mb.3') # Get info

ring.get('k0.mb')# Get value of variable
ring.get('mb.3')# Get raw element
ring.get("k0.mb/2") # Error because it does not exists

ring.eval('k0.mb/2')# Evaluate string
ring.get_expr("k0.mb")# Get exprression associated to var
ring.new_expr("k0.mb/2")# Return new expression

## View
ring["mb.3"].info() ## print table
ring["mb.3"].info('k0') ## print info

ring["mb.3"].get_value('knl') # get the array
ring["mb.3"].get_expr('k0') ## get expresssion
ring["mb.3"].get('knl') # get the raw array  ##PROPOSAL

ring["mb.3"].get_value() # get a dictionary
ring["mb.3"].get_expr() # get dictionary of expressions
ring["mb.3"].to_dict() # get a dictionary ##PROPOSAL
ring["mb.3"].to_expr_dict() ## get a dictionary ##PROPOSAL


ring["mb.3"].get('knl') # get the array
ring["mb.3"].knl # get view the array

"""
Line     :        vars, ref, element_dict, ref_manager,  get_table
Env      : lines, vars, ref, element_dict, ref_manager
Env, Line: get, set, eval, get_expr, new_expr, info

vars     : get, set, eval, get_expr, new_expr, info, get_table
view     : get_expr, get_value, get_info, get_table
"""
