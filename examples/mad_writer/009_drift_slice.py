import xtrack as xt

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=1e9)

line = env.new_line(length=10, components=[
    env.new('q1', xt.Quadrupole, length=1, k1=0.3, at=4),
    env.new('q2', xt.Quadrupole, length=1, k1=-0.3, at=6)
])

line.insert('m', xt.Marker(), at=2)

tw = line.twiss4d()

tw_ng = line.madng_twiss()