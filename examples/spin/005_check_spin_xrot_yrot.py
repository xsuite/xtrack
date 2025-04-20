import xtrack as xt

env = xt.Environment()
env.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=700e9,
    anomalous_magnetic_moment=0.00115965218128
)

line = env.new_line(length=10.5, components=[
    env.new('yrot', xt.YRotation, angle=1, at=5),
    env.new('myrot', xt.YRotation, angle=-1, at=10),
])

tw = line.twiss(spin=True,
           betx=10,
           bety=10,
           px=0.001,
           spin_x=0.001)

