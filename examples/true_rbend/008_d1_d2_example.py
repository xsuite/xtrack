import xtrack as xt

env = xt.Environment()
env.vars.default_to_zero = True
line = env.new_line(length=10., components=[
    env.new('d1a', 'RBend', length=1.0, k0='k0d1a', anchor='start', at=1.),
    env.new('d1b', 'RBend', length=1.0, k0='k0d1b', anchor='start',   at=3.),
    env.new('d2',  'RBend', length=1.0, k0='k0d2',  anchor='start', at=8.),
    env.new('end', 'Marker', at=10.),
])
line.set_particle_ref('proton', p0c=1e9)

l_thick =line.copy(shallow=True)
line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=xt.Uniform(6, mode='thick'))
        ])

env['k0d1a'] = 'k0d1'
env['k0d1b'] = 'k0d1'

opt = line.match(
    solve=False,
    betx=1, bety=1,
    vary=[xt.VaryList(['k0d1', 'k0d2'], step=1e-5)],
    targets=xt.TargetSet(x=0.1, px=0.0, at='end'),
)
opt.solve()

tw = line.twiss(betx=1, bety=1)
tw.plot('x')

