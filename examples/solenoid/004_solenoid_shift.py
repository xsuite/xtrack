import xtrack as xt
import numpy as np
import xobjects as xo

# TODO:
# - field for radiation and spin

env = xt.Environment()
env.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=20e9)

line_ref = env.new_line(components=[
    env.new('sol_ref',xt.UniformSolenoid, length=3, ks=0.2),
    ])
line_ref.cut_at_s(np.linspace(0, 3, 5))
tw_ref = line_ref.twiss(x=0.1, y=0.2, betx=1, bety=1)


x0 = 0.05
y0 = 0.15
line_test = env.new_line(components=[
    env.new('solt_test', xt.UniformSolenoid, length=3, ks=0.2, x0=x0, y0=y0),
    ])
line_test.cut_at_s(np.linspace(0, 3, 5))
tw_test = line_test.twiss(x=x0 + 0.1, y=y0 + 0.2, betx=1, bety=1)

xo.assert_near(tw_test.x, tw_ref.x + x0, rtol=0, atol=1e-14)


