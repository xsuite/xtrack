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
env.new('corrector', xt.Multipole, knl=[0], length=0.1)

girder = env.new_line(components=[
    env.place('mq', at=1),
    env.place('ms', at=0.8, from_='mq'),
    env.place('corrector', at=-0.8, from_='mq'),
])

girder_f = girder.clone(name='f')
girder_d = girder.clone(name='d', mirror=True)
env.set('mq.f', k1='kqf')
env.set('mq.d', k1='kqd')

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


hcell_left = halfcell.replicate(name='l', mirror=True)
hcell_right = halfcell.replicate(name='r')

cell = env.new_line(components=[
    env.new('start', xt.Marker),
    hcell_left,
    hcell_right,
    env.new('end', xt.Marker),
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

opt = cell_ss.match(
    solve=False,
    method='4d',
    vary=xt.VaryList(['kqf.ss', 'kqd.ss'], step=1e-5),
    targets=xt.TargetSet(
        betx=tw_cell.betx[-1], bety=tw_cell.bety[-1], at='start.ss',
    ))
opt.solve()


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

## Insertion

env.vars({
    'k1.q1': 0.025,
    'k1.q2': -0.025,
    'k1.q3': 0.025,
    'k1.q4': -0.02,
    'k1.q5': 0.025,
})

half_insertion = env.new_line(name='half_insertion', components=[

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


# # Check buffer behavior
ring2_sliced = ring2.select()
ring2_sliced.cut_at_s(np.arange(0, ring2.get_length(), 0.5))

env['ring'] = ring2
env['ring_sliced'] = ring2_sliced

def _env_to_dict(self, include_var_management=True):

    out = {}
    out["elements"] = {k: el.to_dict() for k, el in self.element_dict.items()}

    if self.particle_ref is not None:
        out['particle_ref'] = self.particle_ref.to_dict()
    if self._var_management is not None and include_var_management:
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            raise ValueError('The line is part ot a MultiLine object. '
                'To save without expressions please use '
                '`line.to_dict(include_var_management=False)`.\n'
                'To save also the deferred expressions please save the '
                'entire multiline.\n ')

        out.update(self._var_management_to_dict())

    out['lines'] = {}

    for nn, ll in self.lines.items():
        out['lines'][nn] = ll.to_dict(include_element_dict=False,
                                      include_var_management=False)

    return out


@classmethod
def _env_from_dict(cls, dct):
    cls = xt.Environment

    ldummy = xt.Line.from_dict(dct)
    out = cls(element_dict=ldummy.element_dict, particle_ref=ldummy.particle_ref,
            _var_management=ldummy._var_management)
    out._line_vars = xt.line.LineVars(out)

    for nn in dct['lines'].keys():
        ll = xt.Line.from_dict(dct['lines'][nn], env=out, verbose=False)
        out[nn] = ll

    return out


xt.Environment._var_management_to_dict = xt.Line._var_management_to_dict
xt.Environment.to_dict = _env_to_dict
xt.Environment.from_dict = _env_from_dict

edct = env.to_dict()
env2 = xt.Environment.from_dict(edct)

assert env2.element_dict is not env.element_dict

assert np.all(np.array(list(env2.lines.keys()))
            == np.array(list(env.lines.keys())))

for nn, ll in env2.lines.items():
    tt = ll.get_table()
    tt_ref = env[nn].get_table()
    assert np.all(tt.name == tt_ref.name)

# Check that no element is in common
assert len(
    set(env2.element_dict.values()).intersection(set(env.element_dict.values()))
    ) == 0

assert env._xdeps_vref.owner is not env2._xdeps_vref.owner

tw2 = env2['ring'].twiss4d()
tw_ref = env2['ring'].twiss4d()

import xobjects as xo
xo.assert_allclose(tw2.betx, tw_ref.betx, rtol=0, atol=1e-12)
xo.assert_allclose(tw2.bety, tw_ref.bety, rtol=0, atol=1e-12)
xo.assert_allclose(tw2.wx_chrom, tw_ref.wx_chrom, rtol=0, atol=1e-12)
xo.assert_allclose(tw2.wy_chrom, tw_ref.wy_chrom, rtol=0, atol=1e-12)

# In the second environment match betx = 2 * bety
opt = env2['half_insertion'].match(
    solve=False,
    betx=tw_arc.betx[0], bety=tw_arc.bety[0],
    alfx=tw_arc.alfx[0], alfy=tw_arc.alfy[0],
    init_at='e.insertion',
    start='ip', end='e.insertion',
    vary=xt.VaryList(['k1.q1', 'k1.q2', 'k1.q3', 'k1.q4'], step=1e-5),
    targets=[
        xt.TargetSet(alfx=0, alfy=0, at='ip'),
        xt.Target(lambda tw: tw.betx[0] - 2*tw.bety[0], 0),
        xt.Target(lambda tw: tw.betx.max(), xt.LessThan(400)),
        xt.Target(lambda tw: tw.bety.max(), xt.LessThan(400)),
        xt.Target(lambda tw: tw.betx.min(), xt.GreaterThan(2)),
        xt.Target(lambda tw: tw.bety.min(), xt.GreaterThan(2)),
    ]
)
opt.solve()

# Check matched betx=2*bety in env2 but not in env
xo.assert_allclose(env2['ring'].twiss4d()['betx', 'ip.l'],
                   2*env2['ring'].twiss4d()['bety', 'ip.l'],
                   rtol=1e-4, atol=0)
xo.assert_allclose(env2['ring_sliced'].twiss4d()['betx', 'ip.l'],
                   2*env2['ring_sliced'].twiss4d()['bety', 'ip.l'],
                   rtol=1e-4, atol=0)

xo.assert_allclose(env['ring'].twiss4d()['betx', 'ip.l'],
                   env['ring'].twiss4d()['bety', 'ip.l'],
                   rtol=1e-4, atol=0)
xo.assert_allclose(env['ring_sliced'].twiss4d()['betx', 'ip.l'],
                   env['ring_sliced'].twiss4d()['bety', 'ip.l'],
                   rtol=1e-4, atol=0)

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


