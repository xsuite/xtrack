import logging

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import xobjects as xo
import xpart as xp
from scipy.spatial import ConvexHull


plt.close('all')

n_part=10000
ctx = xo.context_default
buf = ctx.new_buffer()

logger = logging.getLogger('xtrack')
logger.setLevel(logging.DEBUG)

# Define aper_0
#aper_0 = xt.LimitRect(_buffer=buf, min_y=-1e-2, max_y=1e-2,
#                                   min_x=-2e-2, max_x=2e-2)
aper_0 = xt.LimitEllipse(_buffer=buf, a=2e-2, b=1e-2)
shift_aper_0 = (1e-2, 0.5e-2)
rot_deg_aper_0 = 10.

# Define aper_1
#aper_1 = xt.LimitEllipse(_buffer=buf, a=1e-2, b=2e-2)
aper_1 = xt.LimitRect(_buffer=buf, min_x=-1e-2, max_x=1e-2,
                                   min_y=-2e-2, max_y=2e-2)
shift_aper_1 = (-5e-3, 1e-2)
rot_deg_aper_1 = 10.


# aper_0_sandwitch
trk_aper_0 = xt.Tracker(_buffer=buf, line=xt.Line(
    elements=[xt.XYShift(_buffer=buf, dx=shift_aper_0[0], dy=shift_aper_0[1]),
              xt.SRotation(_buffer=buf, angle=rot_deg_aper_0),
              aper_0,
              xt.Multipole(_buffer=buf, knl=[0.001]),
              xt.SRotation(_buffer=buf, angle=-rot_deg_aper_0),
              xt.XYShift(_buffer=buf, dx=-shift_aper_0[0], dy=-shift_aper_0[1])]))

# aper_1_sandwitch
trk_aper_1 = xt.Tracker(_buffer=buf, line=xt.Line(
    elements=[xt.XYShift(_buffer=buf, dx=shift_aper_1[0], dy=shift_aper_1[1]),
              xt.SRotation(_buffer=buf, angle=rot_deg_aper_1),
              aper_1,
              xt.Multipole(_buffer=buf, knl=[0.001]),
              xt.SRotation(_buffer=buf, angle=-rot_deg_aper_1),
              xt.XYShift(_buffer=buf, dx=-shift_aper_1[0], dy=-shift_aper_1[1])]))

# Build example line
tracker = xt.Tracker(_buffer=buf, line=xt.Line(
    elements = ((xt.Drift(_buffer=buf, length=0.5),)
                + trk_aper_0.line.elements
                + (xt.Drift(_buffer=buf, length=1),
                   xt.Multipole(_buffer=buf, knl=[1e-3]),
                   xt.Drift(_buffer=buf, length=1),
                   xt.Cavity(_buffer=buf, voltage=3e6, frequency=400e6),
                   xt.Drift(_buffer=buf, length=1.),)
                + trk_aper_1.line.elements)))
num_elements = len(tracker.line.elements)

# Test on full line
particles = xp.Particles(_context=ctx,
            px=np.random.uniform(-0.01, 0.01, 10000),
            py=np.random.uniform(-0.01, 0.01, 10000))

tracker.track(particles)


loss_loc_refinement = xt.LossLocationRefinement(tracker,
                                            n_theta = 360,
                                            r_max = 0.5, # m
                                            dr = 50e-6,
                                            ds = 0.1,
                                            save_refine_trackers=True)

import time
t0 = time.time()

loss_loc_refinement.refine_loss_location(particles)

t1 = time.time()
print(f'Took\t{(t1-t0)*1e3:.2f} ms')

# Visualize apertures
interp_tracker = loss_loc_refinement.refine_trackers[
                                    loss_loc_refinement.i_apertures[1]]
s0 = interp_tracker.s0
s1 = interp_tracker.s1
polygon_0 = interp_tracker.line.elements[0]
polygon_1 = interp_tracker.line.elements[-1]
for ii, (trkr, poly) in enumerate(
                         zip([trk_aper_0, trk_aper_1],
                             [polygon_0, polygon_1])):
    part_gen_range = 0.05
    pp = xp.Particles(
                    p0c=6500e9,
                    x=np.random.uniform(-part_gen_range, part_gen_range, n_part),
                    y=np.random.uniform(-part_gen_range, part_gen_range, n_part))
    x0 = pp.x.copy()
    y0 = pp.y.copy()

    trkr.track(pp)
    ids = pp.particle_id


    fig = plt.figure(ii+1)
    ax = fig.add_subplot(111)
    plt.plot(x0, y0, '.', color='red')
    plt.axis('equal')
    plt.grid(linestyle=':')
    plt.plot(x0[ids][pp.state>0], y0[ids][pp.state>0], '.', color='green')
    plt.plot(poly.x_closed, poly.y_closed, '-k', linewidth=1)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(
        polygon_0.x_closed,
        polygon_0.y_closed,
        s0+polygon_0.x_closed*0,
        color='k', linewidth=3)
ax.plot3D(
        polygon_1.x_closed,
        polygon_1.y_closed,
        s1+polygon_1.x_closed*0,
        color='k', linewidth=3)
s_check = []
for ee, ss in zip(interp_tracker.line.elements,
                  interp_tracker.line.get_s_elements()):
    if ee.__class__ is xt.LimitPolygon:
        ax.plot3D(
                ee.x_closed,
                ee.y_closed,
                s0+ss+0*ee.x_closed,
                alpha=0.9,
                )
        s_check.append(ss)
mask = particles.at_element == loss_loc_refinement.i_apertures[1]
ax.plot3D(particles.x[mask], particles.y[mask], particles.s[mask],
          '.r', markersize=.5)
ax.view_init(65, 62); plt.draw()
plt.show()
