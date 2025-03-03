import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=45.6e9, mass0=xt.ELECTRON_MASS_EV)

l1 = env.new_line(components=[
    env.new('b1', 'Bend', length=0.1)])
p0 = l1.build_particles(x=3e-3)
p = p0.copy()
l1.track(p)
xo.assert_allclose(p.px, 0, rtol=0, atol=1e-15)
env['b1'].knl[2] = 0.1
p = p0.copy()
l1.track(p)
assert np.abs(p.px[0]) > 1e-7

l2 = env.new_line(components=[
    env.new('q1', 'Quadrupole', length=0.1)])
p = p0.copy()
l2.track(p)
xo.assert_allclose(p.px, 0, rtol=0, atol=1e-15)
env['q1'].knl[2] = 0.1
p = p0.copy()
l2.track(p)
assert np.abs(p.px[0]) > 1e-7

l3 = env.new_line(components=[
    env.new('s1', 'Sextupole', length=0.1)])
p = p0.copy()
l3.track(p)
xo.assert_allclose(p.px, 0, rtol=0, atol=1e-15)
env['s1'].knl[2] = 0.1
p = p0.copy()
l3.track(p)
assert np.abs(p.px[0]) > 1e-7

l4 = env.new_line(components=[
    env.new('o1', 'Octupole', length=0.1)])
p = p0.copy()
l4.track(p)
xo.assert_allclose(p.px, 0, rtol=0, atol=1e-15)
env['o1'].knl[2] = 0.1
p = p0.copy()
l4.track(p)
assert np.abs(p.px[0]) > 1e-7

l5 = env.new_line(components=[
    env.new('rb1', 'RBend', length=0.1)])
p0 = l5.build_particles(x=3e-3)
p = p0.copy()
l5.track(p)
xo.assert_allclose(p.px, 0, rtol=0, atol=1e-15)
env['rb1'].knl[2] = 0.1
p = p0.copy()
l5.track(p)
assert np.abs(p.px[0]) > 1e-7
