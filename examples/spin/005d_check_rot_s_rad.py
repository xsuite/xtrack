import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=700e9,
    anomalous_magnetic_moment=0.00115965218128
)

line = env.new_line(length=1., components=[
    env.new('m', xt.Magnet, k0=0.1, length=0.2, rot_s_rad=0.1)])

line_ref = env.new_line(length=1., components=[
    env.new('srot', xt.SRotation, angle=np.rad2deg(0.1)),
    env.new('mref', xt.Magnet, k0=0.1, length=0.2),
    env.new('msrot', xt.SRotation, angle=np.rad2deg(-0.1))])

tw = line.twiss(spin=True,
            betx=10,
            bety=10,
            px=0.1,
            spin_x=0.1)
tw_ref = line_ref.twiss(spin=True,
            betx=10,
            bety=10,
            px=0.1,
            spin_x=0.1)

xo.assert_allclose(tw.px[-1], tw_ref.px[-1], rtol=0, atol=1e-9)
xo.assert_allclose(tw.spin_x[-1], tw_ref.spin_x[-1], rtol=0, atol=1e-9)
xo.assert_allclose(tw.py[-1], tw_ref.py[-1], rtol=0, atol=1e-9)
xo.assert_allclose(tw.spin_y[-1], tw_ref.spin_y[-1], rtol=0, atol=1e-9)