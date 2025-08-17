import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=700e9,
    anomalous_magnetic_moment=0.00115965218128
)

line = env.new_line(length=1., components=[
    env.new('srot', xt.SRotation, angle=12, at=0.2),
    env.new('msrot', xt.SRotation, angle=-12, at=0.4),
])

tw = line.twiss(spin=True,
           betx=10,
           bety=10,
           px=0.1,
           spin_x=0.1)

assert np.all(tw.name == np.array(
    ['drift_1', 'srot', 'drift_2', 'msrot', 'drift_3', '_end_point']))
xo.assert_allclose(tw.s, np.array([0. , 0.2, 0.2, 0.4, 0.4, 1. ]), rtol=0, atol=1e-10)
xo.assert_allclose(tw.py, np.array(
    [ 0.,  0., -2.07911691e-02, -2.07911691e-02, 0.,  0.]), rtol=0, atol=1e-9)
xo.assert_allclose(tw.spin_y, np.array(
    [ 0.,  0., -2.07911691e-02, -2.07911691e-02, 0.,  0.]), rtol=0, atol=1e-9)
xo.assert_allclose(tw.spin_z, 0, rtol=0, atol=1e-9)
xo.assert_allclose(tw.spin_x**2 + tw.spin_y**2 + tw.spin_z**2, 0.1**2,
    rtol=0, atol=1e-9)

tw = line.twiss(spin=True,
           betx=10,
           bety=10,
           spin_z=1.)

xo.assert_allclose(tw.spin_y, 0., rtol=0, atol=1e-9)
xo.assert_allclose(tw.spin_x, 0., rtol=0, atol=1e-9)
xo.assert_allclose(tw.spin_z, 1., rtol=0, atol=1e-9)

tw = line.twiss(spin=True,
              betx=10,
              bety=10,
              spin_y=0.1)

xo.assert_allclose(tw.spin_x,
                   np.array([ 0.,  0., 2.07911691e-02, 2.07911691e-02, 0.,  0.]),
                   rtol=0, atol=1e-9)
xo.assert_allclose(tw.spin_z, 0, rtol=0, atol=1e-9)
xo.assert_allclose(tw.spin_x**2 + tw.spin_y**2 + tw.spin_z**2, 0.1**2,
    rtol=0, atol=1e-9)