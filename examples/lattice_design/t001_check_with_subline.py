import xtrack as xt
import xobjects as xo
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
env.new('corrector', xt.Multipole, knl=[0], length=0.1)

girder = env.new_line(components=[
    env.place('mq', at=1),
    env.place('ms', at=0.8, from_='mq'),
    env.place('corrector', at=-0.8, from_='mq'),
])

tt_girder = girder.get_table(attr=True)
assert np.all(tt_girder.name == np.array(
    ['drift_1', 'corrector', 'drift_2', 'mq', 'drift_3', 'ms', '_end_point']))
tt_girder['s_center'] = tt_girder['s'] + \
    tt_girder['length']/2 * np.float64(tt_girder['isthick'])
xo.assert_allclose(tt_girder['s_center', 'mq'], 1., atol=1e-14, rtol=0)
xo.assert_allclose(tt_girder['s_center', 'ms'] - tt_girder['s_center', 'mq'], 0.8,
                   atol=1e-14, rtol=0)
xo.assert_allclose(
    tt_girder['s_center', 'corrector'] - tt_girder['s_center', 'mq'], -0.8,
    atol=1e-14, rtol=0)


girder_f = girder.clone(name='f')
girder_d = girder.clone(name='d', mirror=True)
env.set('mq.f', k1='kqf')
env.set('mq.d', k1='kqd')

# Check clone
tt_girder_f = girder_f.get_table(attr=True)
assert (~(tt_girder_f.isreplica)).all()
assert np.all(tt_girder_f.name == np.array(
    ['drift_1.f', 'corrector.f', 'drift_2.f', 'mq.f', 'drift_3.f', 'ms.f', '_end_point']))
tt_girder_f['s_center'] = (tt_girder_f['s']
                           + tt_girder_f['length']/2 * np.float64(tt_girder_f['isthick']))
xo.assert_allclose(tt_girder_f['s_center', 'mq.f'], 1., atol=1e-14, rtol=0)
xo.assert_allclose(tt_girder_f['s_center', 'ms.f'] - tt_girder_f['s_center', 'mq.f'], 0.8,
                   atol=1e-14, rtol=0)
xo.assert_allclose(
    tt_girder_f['s_center', 'corrector.f'] -
    tt_girder_f['s_center', 'mq.f'], -0.8,
    atol=1e-14, rtol=0)

# Check clone mirror
tt_girder_d = girder_d.get_table(attr=True)
assert (~(tt_girder_d.isreplica)).all()
len_girder = tt_girder_d.s[-1]
assert np.all(tt_girder_d.name == np.array(
    ['ms.d', 'drift_3.d', 'mq.d', 'drift_2.d', 'corrector.d', 'drift_1.d', '_end_point']))
tt_girder_d['s_center'] = (tt_girder_d['s']
                           + tt_girder_d['length']/2 * np.float64(tt_girder_d['isthick']))
xo.assert_allclose(tt_girder_d['s_center', 'mq.d'],
                   len_girder - 1., atol=1e-14, rtol=0)
xo.assert_allclose(tt_girder_d['s_center', 'ms.d'] - tt_girder_d['s_center', 'mq.d'],
                   -0.8, atol=1e-14, rtol=0)
xo.assert_allclose(tt_girder_d['s_center', 'corrector.d'] - tt_girder_d['s_center', 'mq.d'],
                   0.8, atol=1e-14, rtol=0)


halfcell = env.new_line(components=[

    # End of the half cell (will be mid of the cell)
    env.new('mid', xt.Marker, at='l.halfcell'),

    # Bends
    env.new('mb.2', 'mb', at='l.halfcell / 2'),
    env.new('mb.1', 'mb', at='-l.mb - 1', from_='mb.2'),
    env.new('mb.3', 'mb', at='l.mb + 1', from_='mb.2'),

    # Quadrupoles, sextupoles and correctors
    env.place(girder_d, at=1.2),
    env.place(girder_f, at='l.halfcell - 1.2'),

])

l_hc = env.vv['l.halfcell']
xo.assert_allclose(l_hc, l_hc, atol=1e-14, rtol=0)
tt_hc = halfcell.get_table(attr=True)
assert np.all(tt_hc.name == np.array(
    ['drift_4', 'ms.d', 'drift_3.d', 'mq.d', 'drift_2.d', 'corrector.d',
     'drift_1.d', 'drift_5', 'mb.1', 'drift_6', 'mb.2', 'drift_7',
     'mb.3', 'drift_8', 'drift_1.f', 'corrector.f', 'drift_2.f', 'mq.f',
     'drift_3.f', 'ms.f', 'drift_9', 'mid', '_end_point']))
assert np.all(tt_hc.element_type == np.array(
    ['Drift', 'Sextupole', 'Drift', 'Quadrupole', 'Drift', 'Multipole',
     'Drift', 'Drift', 'Bend', 'Drift', 'Bend', 'Drift', 'Bend',
     'Drift', 'Drift', 'Multipole', 'Drift', 'Quadrupole', 'Drift',
     'Sextupole', 'Drift', 'Marker', '']))
assert np.all(tt_hc.isreplica == False)
tt_hc['s_center'] = (
    tt_hc['s'] + tt_hc['length'] / 2 * np.float64(tt_hc['isthick']))
xo.assert_allclose(tt_hc['s_center', 'mq.d'],
                   1.2 - tt_girder_d.s[-1] / 2 +
                   tt_girder_d['s_center', 'mq.d'],
                   atol=1e-14, rtol=0)
xo.assert_allclose(tt_hc['s_center', 'ms.f'] - tt_hc['s_center', 'mq.f'], 0.8,
                   atol=1e-14, rtol=0)
xo.assert_allclose(
    tt_hc['s_center', 'corrector.f'] - tt_hc['s_center', 'mq.f'], -0.8,
    atol=1e-14, rtol=0)
xo.assert_allclose(tt_hc['s_center', 'ms.d'] - tt_hc['s_center', 'mq.d'],
                   -0.8, atol=1e-14, rtol=0)
xo.assert_allclose(tt_hc['s_center', 'corrector.d'] - tt_hc['s_center', 'mq.d'],
                   0.8, atol=1e-14, rtol=0)
xo.assert_allclose(tt_hc['s_center', 'mb.2'], l_hc / 2, atol=1e-14, rtol=0)
xo.assert_allclose(tt_hc['s_center', 'mb.1'], tt_hc['s_center', 'mb.2'] - env.vv['l.mb'] - 1,
                     atol=1e-14, rtol=0)
xo.assert_allclose(tt_hc['s_center', 'mb.3'], tt_hc['s_center', 'mb.2'] + env.vv['l.mb'] + 1,
                   atol=1e-14, rtol=0)


hcell_left = halfcell.replicate(name='l', mirror=True)
hcell_right = halfcell.replicate(name='r')

cell = env.new_line(components=[
    env.new('start', xt.Marker),
    hcell_left,
    hcell_right,
    env.new('end', xt.Marker),
])

tt_cell = cell.get_table(attr=True)
tt_cell['s_center'] = (
    tt_cell['s'] + tt_cell['length'] / 2 * np.float64(tt_cell['isthick']))
assert np.all(tt_cell.name == np.array(
    ['start', 'mid.l', 'drift_9.l', 'ms.f.l', 'drift_3.f.l', 'mq.f.l',
      'drift_2.f.l', 'corrector.f.l', 'drift_1.f.l', 'drift_8.l',
      'mb.3.l', 'drift_7.l', 'mb.2.l', 'drift_6.l', 'mb.1.l',
      'drift_5.l', 'drift_1.d.l', 'corrector.d.l', 'drift_2.d.l',
      'mq.d.l', 'drift_3.d.l', 'ms.d.l', 'drift_4.l', 'drift_4.r',
      'ms.d.r', 'drift_3.d.r', 'mq.d.r', 'drift_2.d.r', 'corrector.d.r',
      'drift_1.d.r', 'drift_5.r', 'mb.1.r', 'drift_6.r', 'mb.2.r',
      'drift_7.r', 'mb.3.r', 'drift_8.r', 'drift_1.f.r', 'corrector.f.r',
      'drift_2.f.r', 'mq.f.r', 'drift_3.f.r', 'ms.f.r', 'drift_9.r',
      'mid.r', 'end', '_end_point']))
assert np.all(tt_cell.element_type == np.array(
    ['Marker', 'Marker', 'Drift', 'Sextupole', 'Drift', 'Quadrupole',
       'Drift', 'Multipole', 'Drift', 'Drift', 'Bend', 'Drift', 'Bend',
       'Drift', 'Bend', 'Drift', 'Drift', 'Multipole', 'Drift',
       'Quadrupole', 'Drift', 'Sextupole', 'Drift', 'Drift', 'Sextupole',
       'Drift', 'Quadrupole', 'Drift', 'Multipole', 'Drift', 'Drift',
       'Bend', 'Drift', 'Bend', 'Drift', 'Bend', 'Drift', 'Drift',
       'Multipole', 'Drift', 'Quadrupole', 'Drift', 'Sextupole', 'Drift',
       'Marker', 'Marker', '']))
assert np.all(tt_cell.isreplica == np.array(
    [False,  True,  True,  True,  True,  True,  True,  True,  True,
      True,  True,  True,  True,  True,  True,  True,  True,  True,
      True,  True,  True,  True,  True,  True,  True,  True,  True,
      True,  True,  True,  True,  True,  True,  True,  True,  True,
      True,  True,  True,  True,  True,  True,  True,  True,  True,
      False, False]))

tt_cell_stripped = tt_cell.rows[1:-2] # Remove _end_point and markers added in cell
tt_cell_second_half = tt_cell_stripped.rows[len(tt_cell_stripped)//2 :]
tt_cell_second_half.s_center -= tt_cell_second_half.s[0]
tt_hc_stripped = tt_hc.rows[:-1] # Remove _end_point
xo.assert_allclose(tt_cell_second_half.s_center, tt_hc_stripped.s_center, atol=5e-14, rtol=0)
tt_cell_first_half = tt_cell_stripped.rows[:len(tt_cell_stripped)//2]
s_center_mirrored_first_half = (
    tt_cell_stripped['s', len(tt_cell_stripped)//2] - tt_cell_first_half.s_center[::-1])
xo.assert_allclose(s_center_mirrored_first_half, tt_hc_stripped.s_center, atol=5e-14, rtol=0)

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

hcell_left_ss = halfcell_ss.replicate(name='l', mirror=True)
hcell_right_ss = halfcell_ss.replicate(name='r')
cell_ss = env.new_line(components=[
    env.new('start.ss', xt.Marker),
    hcell_left_ss,
    hcell_right_ss,
    env.new('end.ss', xt.Marker),
])


arc = env.new_line(components=[
    cell.replicate(name='cell.1'),
    cell.replicate(name='cell.2'),
    cell.replicate(name='cell.3'),
])

assert 'cell.2' in env.lines
tt_cell2 = env.lines['cell.2'].get_table(attr=True)
assert np.all(tt_cell2.name[:-1] == np.array([
    nn+'.cell.2' for nn in tt_cell.name[:-1]]))
assert np.all(tt_cell2.s == tt_cell.s)
assert tt_cell2.isreplica[:-1].all()
assert tt_cell2['parent_name', 'mq.d.l.cell.2'] == 'mq.d.l'
assert tt_cell2['parent_name', 'mq.f.l.cell.2'] == 'mq.f.l'
assert tt_cell['parent_name', 'mq.d.l'] == 'mq.d'
assert tt_cell['parent_name', 'mq.f.l'] == 'mq.f'

tt_arc = arc.get_table(attr=True)
assert len(tt_arc) == 3 * (len(tt_cell)-1) + 1
n_cell = len(tt_cell) - 1
assert np.all(tt_arc.name[n_cell:2*n_cell] == tt_cell2.name[:-1])
for nn in tt_cell2.name[:-1]:
    assert arc.get(nn) is env.get(nn)
    assert arc.get(nn) is env['cell.2'].get(nn)

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
tt_ring = ring.get_table(attr=True)
# Check length
xo.assert_allclose(tt_ring.s[-1], 2*l_hc * (n_cells_par_arc * n_arcs + 2*n_arcs),
                     atol=1e-12, rtol=0)
# Check closure
sv_ring = ring.survey()
xo.assert_allclose(sv_ring.X[-1], 0, atol=1e-12, rtol=0)
xo.assert_allclose(sv_ring.Y[-1], 0, atol=1e-12, rtol=0)
xo.assert_allclose(sv_ring.Z[-1], 0, atol=1e-12, rtol=0)

xo.assert_allclose(sv_ring.angle.sum(), 2*np.pi, atol=1e-12, rtol=0)

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

insertion = env.new_line([
    half_insertion.replicate('l', mirror=True),
    half_insertion.replicate('r')])

ring2 = env.new_line(components=[
    env['arc.1'],
    env['ss.1'],
    env['arc.2'],
    insertion,
    env['arc.3'],
    env['ss.3'],
])


# # Check buffer behavior
ring2_sliced = ring2.select()
ring2_sliced.cut_at_s(np.arange(0, ring2.get_length(), 0.5))

opt = cell.match(
    method='4d',
    vary=xt.VaryList(['kqf', 'kqd'], step=1e-5),
    targets=xt.TargetSet(
        qx=0.333333,
        qy=0.333333,
    ))
tw_cell = cell.twiss4d()

opt = cell_ss.match(
    solve=False,
    method='4d',
    vary=xt.VaryList(['kqf.ss', 'kqd.ss'], step=1e-5),
    targets=xt.TargetSet(
        betx=tw_cell.betx[-1], bety=tw_cell.bety[-1], at='start.ss',
    ))
opt.solve()

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

# Check that the cell is matched to the rest of the ring
tw = ring.twiss4d()
tw_cell_from_ring = tw.rows['start.cell.3.arc.2':'end.cell.3.arc.2']
xo.assert_allclose(tw_cell_from_ring.betx, tw_cell.betx[:-1], atol=0, rtol=2e-4)
xo.assert_allclose(tw_cell_from_ring.bety, tw_cell.bety[:-1], atol=0, rtol=2e-4)

tw2 = ring2.twiss4d()
tw_cell_from_ring2 = tw2.rows['start.cell.3.arc.2':'end.cell.3.arc.2']
xo.assert_allclose(tw_cell_from_ring2.betx, tw_cell.betx[:-1], atol=0, rtol=2e-4)
xo.assert_allclose(tw_cell_from_ring2.bety, tw_cell.bety[:-1], atol=0, rtol=2e-4)

# Check select
cell3_select = ring2.select(start='start.cell.3.arc.2', end='end.cell.3.arc.2',
                            name='cell3_copy')
assert 'cell3_copy' in env.lines
assert env.lines['cell3_copy'] is cell3_select
assert cell3_select._element_dict is env.element_dict
assert cell3_select.element_names[0] == 'start.cell.3.arc.2'
assert cell3_select.element_names[-1] == 'end.cell.3.arc.2'
assert (np.array(cell3_select.element_names) == np.array(
    tw.rows['start.cell.3.arc.2':'end.cell.3.arc.2'].name)).all()

# Check that they share the _element_dict
assert cell._element_dict is env.element_dict
assert halfcell._element_dict is env.element_dict
assert halfcell_ss._element_dict is env.element_dict
assert cell_ss._element_dict is env.element_dict
assert insertion._element_dict is env.element_dict
assert ring2._element_dict is env.element_dict

cell3_select.twiss4d()

tw2_slice = ring2_sliced.twiss4d()
xo.assert_allclose(tw2_slice['betx', 'ip.l'], tw2['betx', 'ip.l'], atol=0, rtol=2e-4)
xo.assert_allclose(tw2_slice['bety', 'ip.l'], tw2['bety', 'ip.l'], atol=0, rtol=2e-4)
xo.assert_allclose(tw2_slice['alfx', 'ip.l'], 0, atol=1e-6, rtol=0)
xo.assert_allclose(tw2_slice['alfy', 'ip.l'], 0, atol=1e-6, rtol=0)
xo.assert_allclose(tw2_slice['dx', 'ip.l'], 0, atol=1e-4, rtol=0)
xo.assert_allclose(tw2_slice['dpx', 'ip.l'], 0, atol=1e-6, rtol=0)
xo.assert_allclose(tw2_slice['dy', 'ip.l'], 0, atol=1e-4, rtol=0)
xo.assert_allclose(tw2_slice['dpy', 'ip.l'], 0, atol=1e-6, rtol=0)

import matplotlib.pyplot as plt
plt.close('all')
for ii, rr in enumerate([ring, ring2_sliced]):

    ttww = rr.twiss4d()

    fig = plt.figure(ii, figsize=(6.4*1.2, 4.8))
    ax1 = fig.add_subplot(2, 1, 1)
    pltbet = ttww.plot('betx bety', ax=ax1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    pltdx = ttww.plot('dx', ax=ax2)
    fig.subplots_adjust(right=.85)
    pltbet.move_legend(1.2,1)
    pltdx.move_legend(1.2,1)

ring2.survey().plot()


plt.show()



