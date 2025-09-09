import xtrack as xt

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)
line = env.new_line(length=5, components=[
    env.new('q', xt.Quadrupole, k1=0.1, length=1.0, at=1.5),
    env.new('end', xt.Marker, at=5.0)
])

line['q'].rot_shift_anchor=0.5
line['q'].rot_x_rad=0.1

tw = line.twiss(betx=1, bety=2)
print("tw:", tw)
tw_back = line.twiss(init_at='end', init=tw)
print("tw_back:", tw_back)