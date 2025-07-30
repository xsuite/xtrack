import xtrack as xt
import numpy as np

env = xt.Environment(particle_ref=xt.Particles(p0c = 1E9))

line = env.new_line(length=10, components=[
    env.new('b1', 'Bend', angle=0.1,  length=1., at=2),
    env.new('b2', 'Bend', angle=-0.1, length=1., at=4),
    env.new('b3', 'Bend', angle=-0.1, length=1., at=6),
    env.new('b4', 'Bend', angle=0.1,  length=1., at=8),

    env.new('end', 'Marker', at=10)
])

sv = line.survey()
svback = line.survey(element0='end', Z0=sv.Z[-1])

sv.cols['s X Y Z'].show()
svback.cols['s X Y Z'].show()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(sv.Z, sv.X, '.-')
plt.plot(svback.Z, svback.X, 'x-')

plt.show()