import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

from _madpoint import MadPoint 

env = xt.Environment()

env.particle_ref = xt.Particles(p0c = 1E9)
env['k0']   = 0.00
env['h']    = 1E-3

line    = env.new_line(name = 'line', components=[
    env.new('bend', xt.Bend, k0 = 'k0', h = 'h', length = 0.5, at=2),
    env.new('xyshift', xt.XYShift, dx = 0.1, dy=0.2),
    env.new('end', xt.Marker, at = 5)])

line.configure_bend_model(core = 'bend-kick-bend', edge = 'suppressed')
line.cut_at_s(np.linspace(0, line.get_length(), 101))

sv = line.survey()
tw = line.twiss4d(betx = 1, bety  =1)

madpoints = []
xx = []
yy = []
zz = []
for nn in tw.name:
    madpoints.append(
        MadPoint(name = nn, xsuite_twiss = tw, xsuite_survey = sv))
    xx.append(madpoints[-1].p[0])
    yy.append(madpoints[-1].p[1])
    zz.append(madpoints[-1].p[2])

xx = np.array(xx)
yy = np.array(yy)
zz = np.array(zz)

sv['xx'] = xx
sv['yy'] = yy
sv['zz'] = zz


fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(sv.Z, sv.X)
axs[1].plot(tw.s, tw.x)
axs[2].plot(sv.s, sv.xx)

axs[0].set_xlabel('Z [m]')
axs[0].set_ylabel('X [m]')

axs[1].set_xlabel('s [m]')
axs[1].set_ylabel('x [m]')

axs[2].set_xlabel('s [m]')
axs[2].set_ylabel('xx [m]')

plt.show()