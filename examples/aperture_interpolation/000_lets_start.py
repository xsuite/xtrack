import numpy as np

import xline as xl
import xtrack as xt
import xobjects as xo
import xpart as xp

n_part=10000
ctx = xo.context_default
buf = ctx.new_buffer()

aper_0 = xt.LimitEllipse(_buffer=buf, a=2e-2, b=1e-2)
shift_aper_0 = (1e-2, 0.5e-2)
rot_deg_aper_0 = 30.

trk_aper_0 = xt.Tracker(_buffer=buf, sequence=xl.Line(
    elements=[xt.XYShift(_buffer=buf, dx=shift_aper_0[0], dy=shift_aper_0[1]),
              xt.SRotation(_buffer=buf, angle=rot_deg_aper_0),
              aper_0,
              xt.SRotation(_buffer=buf, angle=-rot_deg_aper_0),
              xt.XYShift(_buffer=buf, dx=-shift_aper_0[0], dy=-shift_aper_0[1])]))


part_gen_range = 0.05
pp = xt.Particles(
                p0c=6500e9,
                x=np.random.uniform(-part_gen_range, part_gen_range, n_part),
                y=np.random.uniform(-part_gen_range, part_gen_range, n_part))
x0 = pp.x.copy()
y0 = pp.y.copy()

trk_aper_0.track(pp)
ids = pp.particle_id

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
plt.plot(x0, y0, '.', color='red')
plt.plot(x0[ids][pp.state>0], y0[ids][pp.state>0], '.', color='green')

plt.show()
