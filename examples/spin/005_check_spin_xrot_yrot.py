import xtrack as xt
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=700e9,
    anomalous_magnetic_moment=0.00115965218128
)
breakpoint()
line = env.new_line(length=1., components=[
    env.new('yrot', xt.YRotation, angle=12, at=0.2),
    env.new('myrot', xt.YRotation, angle=-12, at=0.4),
])

tw = line.twiss(spin=True,
           betx=10,
           bety=10,
           px=0.001,
           spin_x=0.001)

assert np.all(tw.name == np.array(
    ['drift_1', 'yrot', 'drift_2', 'myrot', 'drift_3', '_end_point']))