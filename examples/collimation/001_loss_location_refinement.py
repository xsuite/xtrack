# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import logging                            #!skip-doc
import numpy as np

import xtrack as xt
import xobjects as xo
import xpart as xp

# Display debug information .         #!skip-doc
logger = logging.getLogger('xtrack')  #!skip-doc
logger.setLevel(logging.DEBUG)        #!skip-doc

###################
# Build test line #
###################

ctx = xo.context_default
buf = ctx.new_buffer()

# We build a test line having two aperture elements which are shifted and
# rotated w.r.t. the accelerator reference frame.

# Define aper_0
aper_0 = xt.LimitEllipse(_buffer=buf, a=2e-2, b=1e-2)
shift_aper_0 = (1e-2, 0.5e-2)
rot_deg_aper_0 = 10.

# Define aper_1
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

#################
# Build tracker #
#################

line=xt.Line(
    elements = ((xt.Drift(_buffer=buf, length=0.5),)
                + trk_aper_0.line.elements
                + (xt.Drift(_buffer=buf, length=1),
                   xt.Drift(_buffer=buf, length=1),
                   xt.Drift(_buffer=buf, length=1.),)
                + trk_aper_1.line.elements))
line.build_tracker(_buffer=buf)
num_elements = len(line.element_names)

# Generate test particles
particles = xp.Particles(_context=ctx,
            px=np.random.uniform(-0.01, 0.01, 10000),
            py=np.random.uniform(-0.01, 0.01, 10000))

#########
# Track #
#########

line.track(particles)

########################
# Refine loss location #
########################

loss_loc_refinement = xt.LossLocationRefinement(line,
        n_theta = 360, # Angular resolution in the polygonal approximation of the aperture
        r_max = 0.5, # Maximum transverse aperture in m
        dr = 50e-6, # Transverse loss refinement accuracy [m]
        ds = 0.1, # Longitudinal loss refinement accuracy [m]
        save_refine_trackers=True # Diagnostics flag
        )

import time                                              #!skip-doc
t0 = time.time()                                         #!skip-doc

loss_loc_refinement.refine_loss_location(particles)

t1 = time.time()                                         #!skip-doc
print(f'Took\t{(t1-t0)*1e3:.2f} ms')                     #!skip-doc


#!end-doc-part

import matplotlib.pyplot as plt
plt.close('all')

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
                    x=np.random.uniform(-part_gen_range, part_gen_range, 10000),
                    y=np.random.uniform(-part_gen_range, part_gen_range, 10000))
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
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('s [m]')
plt.show()
