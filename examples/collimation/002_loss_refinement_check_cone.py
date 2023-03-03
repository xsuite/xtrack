# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import logging

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import xobjects as xo
import xpart as xp
from scipy.spatial import ConvexHull


plt.close('all')

n_part=10000
shift_x = 0.3e-2
shift_y = 0.5e-2

ctx = xo.context_default
buf = ctx.new_buffer()

logger = logging.getLogger('xtrack')
logger.setLevel(logging.DEBUG)

# Define aper_0
aper_0 = xt.LimitEllipse(_buffer=buf, a=2e-2, b=2e-2)
shift_aper_0 = (shift_x, shift_y)
rot_deg_aper_0 = 10.

# Define aper_1
aper_1 = xt.LimitEllipse(_buffer=buf, a=1e-2, b=1e-2)
shift_aper_1 = (shift_x, shift_y)
rot_deg_aper_1 = 10.

# aper_0_sandwitch
line_aper_0 = xt.Line(
    elements=[xt.XYShift(_buffer=buf, dx=shift_aper_0[0], dy=shift_aper_0[1]),
              xt.SRotation(_buffer=buf, angle=rot_deg_aper_0),
              aper_0,
              xt.Multipole(_buffer=buf, knl=[0.00]),
              xt.SRotation(_buffer=buf, angle=-rot_deg_aper_0),
              xt.XYShift(_buffer=buf, dx=-shift_aper_0[0], dy=-shift_aper_0[1])])
line_aper_0.build_tracker(_buffer=buf)

# aper_1_sandwitch
line_aper_1 = xt.Line(
    elements=[xt.XYShift(_buffer=buf, dx=shift_aper_1[0], dy=shift_aper_1[1]),
              xt.SRotation(_buffer=buf, angle=rot_deg_aper_1),
              aper_1,
              xt.Multipole(_buffer=buf, knl=[0.00]),
              xt.SRotation(_buffer=buf, angle=-rot_deg_aper_1),
              xt.XYShift(_buffer=buf, dx=-shift_aper_1[0], dy=-shift_aper_1[1])])
line_aper_1.build_tracker(_buffer=buf)

# Build example line
line=xt.Line(
    elements = ((xt.Drift(_buffer=buf, length=0.5),)
                + line_aper_0.elements
                + (xt.Drift(_buffer=buf, length=1),
                   xt.Multipole(_buffer=buf, knl=[0.]),
                   xt.Drift(_buffer=buf, length=1),
                   xt.Cavity(_buffer=buf, voltage=3e6, frequency=400e6),
                   xt.Drift(_buffer=buf, length=1.),)
                + line_aper_1.elements))
num_elements = len(line)

line.build_tracker()

# Test on full line
r = np.linspace(0, 0.018, n_part)
theta = np.linspace(0, 8*np.pi, n_part)
particles = xp.Particles(_context=ctx,
        p0c=6500e9,
        x=r*np.cos(theta)+shift_x,
        y=r*np.sin(theta)+shift_y)

line.track(particles)


loss_loc_refinement = xt.LossLocationRefinement(line,
                                            n_theta = 360,
                                            r_max = 0.5, # m
                                            dr = 50e-6,
                                            ds = 0.1,
                                            save_refine_trackers=True,
                                            allowed_backtrack_types=[
                                                xt.Multipole, xt.Cavity])

import time
t0 = time.time()

loss_loc_refinement.refine_loss_location(particles)

t1 = time.time()
print(f'Took\t{(t1-t0)*1e3:.2f} ms')


# Automatic checks
mask_lost = particles.state == 0
r_calc = np.sqrt((particles.x-shift_x)**2 + (particles.y-shift_y)**2)
assert np.all(r_calc[~mask_lost]<1e-2)
assert np.all(r_calc[mask_lost]>1e-2)
i_aper_1 = line.elements.index(aper_1)
assert np.all(particles.at_element[mask_lost]==i_aper_1)
assert np.all(particles.at_element[~mask_lost]==0)
s0 = line.get_s_elements()[line.elements.index(aper_0)]
s1 = line.get_s_elements()[line.elements.index(aper_1)]
r0 = np.sqrt(aper_0.a_squ)
r1 = np.sqrt(aper_1.a_squ)
s_expected = s0 + (r_calc-r0)/(r1 - r0)*(s1 - s0)
# TODO This threshold is a bit large
assert np.allclose(particles.s[mask_lost], s_expected[mask_lost], atol=0.11)

# Visualize apertures
interp_tracker = loss_loc_refinement.refine_trackers[
                                    loss_loc_refinement.i_apertures[1]]
s0 = interp_tracker.s0
s1 = interp_tracker.s1
polygon_0 = interp_tracker.line.elements[0]
polygon_1 = interp_tracker.line.elements[-1]
for ii, (trkr, poly) in enumerate(
                         zip([line_aper_0,line_aper_1],
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
r_check = []
for ee, ss in zip(interp_tracker.line.elements,
                  interp_tracker.line.get_s_elements()):
    if ee.__class__ is xt.LimitPolygon:
        ax.plot3D(
                ee.x_closed,
                ee.y_closed,
                s0+ss+0*ee.x_closed,
                alpha=0.9,
                )
        s_check.append(ss+s0)
        r_vertices = np.sqrt(
                (ee.x_vertices-shift_x)**2 + (ee.y_vertices-shift_y)**2)
        r_check.append(np.mean(r_vertices))
mask = particles.at_element == loss_loc_refinement.i_apertures[1]
ax.plot3D(particles.x[mask], particles.y[mask], particles.s[mask],
          '.r', markersize=.5)
ax.view_init(65, 62); plt.draw()
plt.show()
