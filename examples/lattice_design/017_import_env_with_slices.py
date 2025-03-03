import xtrack as xt

env1 = xt.Environment()
env1.vars.default_to_zero  = True
line1 = env1.new_line(components=[
    env1.new('qq1_thick', xt.Quadrupole, length=1., k1='kk', at=10),
    env1.new('qq1_thin', xt.Quadrupole, length=1., k1='kk', at=20),
    env1.new('qq_shared_thick', xt.Quadrupole, length=1., k1='kk', at=30),
    env1.new('qq_shared_thin', xt.Quadrupole, length=1., k1='kk', at=40),
])
line1.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(5, mode='thick'), name='qq1_thick'),
        xt.Strategy(slicing=xt.Teapot(5, mode='thin'), name='qq1_thin'),
        xt.Strategy(slicing=xt.Teapot(5, mode='thick'), name='qq_shared_thick'),
        xt.Strategy(slicing=xt.Teapot(5, mode='thin'), name='qq_shared_thin'),
    ])

env2 = xt.Environment()
env2.vars.default_to_zero  = True
line2 = env2.new_line(components=[
    env2.new('qq2_thick', xt.Quadrupole, length=1., k1='kk', at=10),
    env2.new('qq2_thin', xt.Quadrupole, length=1., k1='kk', at=20),
    env2.new('qq_shared_thick', xt.Quadrupole, length=1., k1='kk', at=30),
    env2.new('qq_shared_thin', xt.Quadrupole, length=1., k1='kk', at=40),
])

env = xt.Environment(lines={'line1': line1, 'line2': line2})
