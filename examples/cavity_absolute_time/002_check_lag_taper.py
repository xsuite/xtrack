import xtrack as xt
import xobjects as xo

env = xt.Environment()

line = env.new_line(components=[
    env.new('cav', xt.Cavity, frequency=400e6, voltage=16e6),
])
p0 = xt.Particles(p0c=450e9)

p1 = p0.copy()
line.track(p1)
xo.assert_allclose(p1.delta, 0, atol=1e-10)

env['cav'].lag = 60
p2 = p0.copy()
line.track(p2)
assert p2.delta != 0

env['cav'].lag_taper = -60
p3 = p0.copy()
line.track(p3)
# lag_taper should be ignored because radiation is off
xo.assert_allclose(p3.delta, p2.delta, atol=1e-10)

line.configure_radiation(model='mean')
p4 = p0.copy()
line.track(p4)
# lag_taper should be applied now
xo.assert_allclose(p4.delta, 0, atol=1e-10)