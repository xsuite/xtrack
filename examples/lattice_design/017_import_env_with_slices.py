import xtrack as xt

env1 = xt.Environment()
env1.vars.default_to_zero  = True
line1 = env1.new_line(components=[
    env1.new('qq', xt.Quadrupole, length=1., k1='kk')
])
line1.slice_thick_elements(
    slicing_strategies=[
        # Slicing with thin elements
        xt.Strategy(slicing=xt.Teapot(5, mode='thick')), 
    ])

env2 = xt.Environment()
env2.import_line(line=line1, line_name='ll')
env2['ll'].particle_ref = xt.Particles(p0c=7e12)
env2['ll'].twiss(betx=1, bety=1)