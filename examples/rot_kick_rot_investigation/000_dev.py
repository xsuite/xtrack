import xtrack as xt

angle = 0.1
length = 2

env = xt.Environment()

env.elements['r1'] = xt.Bend(length=length, k0=0, angle=angle/2)
env.elements['r2'] = xt.Bend(length=length, k0=0, angle=angle/2)
env.elements['k1'] = xt.Multipole(length=0, knl=[angle])

line = env.new_line(components=['r1', 'k1', 'r2'])
line.set_particle_ref(energy0=1e9)

tw = line.twiss(betx=1, bety=1)
