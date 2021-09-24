import numpy as np
import matplotlib.pyplot as plt

import xline as xl
import xtrack as xt
import xobjects as xo
import xpart as xp
from scipy.spatial import ConvexHull

import aper_interpolation as ap

plt.close('all')

n_part=10000
ctx = xo.context_default
buf = ctx.new_buffer()

pp = xt.LimitPolygon(x_vertices=[1,-1, -1, 1], y_vertices=[1,1,-1,-1])
na = lambda a : np.array(a, dtype=np.float64)
pp.impact_point_and_normal(x_in=na([0]), y_in=na([0]), z_in=na([0]),
                           x_out=na([2]), y_out=na([2]), z_out=na([0]))

# Define aper_0
aper_0 = xt.LimitEllipse(_buffer=buf, a=2e-2, b=1e-2)
shift_aper_0 = (0,0); (1e-2, 0.5e-2)
rot_deg_aper_0 = 10.

# Define aper_1
aper_1 = xt.LimitRect(_buffer=buf, min_x=-1e-2, max_x=1e-2,
                                   min_y=-2e-2, max_y=2e-2)
shift_aper_1 = (0,0); (-5e-3, 1e-2)
rot_deg_aper_1 = 10.


# aper_0_sandwitch
trk_aper_0 = xt.Tracker(_buffer=buf, sequence=xl.Line(
    elements=[xt.XYShift(_buffer=buf, dx=shift_aper_0[0], dy=shift_aper_0[1]),
              xt.SRotation(_buffer=buf, angle=rot_deg_aper_0),
              aper_0,
              xt.Multipole(_buffer=buf, knl=[0.001]),
              xt.SRotation(_buffer=buf, angle=-rot_deg_aper_0),
              xt.XYShift(_buffer=buf, dx=-shift_aper_0[0], dy=-shift_aper_0[1])]))

# aper_1_sandwitch
trk_aper_1 = xt.Tracker(_buffer=buf, sequence=xl.Line(
    elements=[xt.XYShift(_buffer=buf, dx=shift_aper_1[0], dy=shift_aper_1[1]),
              xt.SRotation(_buffer=buf, angle=rot_deg_aper_1),
              aper_1,
              xt.Multipole(_buffer=buf, knl=[0.001]),
              xt.SRotation(_buffer=buf, angle=-rot_deg_aper_1),
              xt.XYShift(_buffer=buf, dx=-shift_aper_1[0], dy=-shift_aper_1[1])]))

# Build example line
tracker = xt.Tracker(_buffer=buf, sequence=xl.Line(
    elements = (trk_aper_0.line.elements
                + (xt.Drift(_buffer=buf, length=1),
                   xt.Drift(_buffer=buf, length=2))
                + trk_aper_1.line.elements)))

# Test on full line
particles = xt.Particles(_context=ctx,
            px=np.random.uniform(-0.01, 0.01, 10000),
            py=np.random.uniform(-0.01, 0.01, 10000))

tracker.track(particles)

# Get backtracker
backtracker = tracker.get_backtracker(_context=ctx)

# Find apertures

i_apertures = []
for ii, ee in enumerate(tracker.line.elements):
    if ee.__class__.__name__.startswith('Limit'):
        i_apertures.append(ii)


i_aper_1 = i_apertures[1]
i_aper_0 = i_apertures[0]

import time
t0 = time.time()

# Get polygons
n_theta = 360 * 5
r_max = 0.5 # m
dr = 50e-6

polygon_1 = ap.characterize_aperture(tracker, i_aper_1, n_theta, r_max, dr)
num_elements = len(tracker.line.elements)
polygon_0 = ap.characterize_aperture(backtracker, num_elements-i_aper_0-1,
                                     n_theta, r_max, dr)

s0 = tracker.line.element_s_locations[i_aper_0]
s1 = tracker.line.element_s_locations[i_aper_1]

# Interpolate

Delta_s = s1 - s0
ds = 0.1

s_vect = np.arange(s0, s1, ds)

interp_polygons = []
for ss in s_vect:
    x_non_convex=(polygon_1.x_vertices*(ss - s0) / Delta_s
              + polygon_0.x_vertices*(s1 - ss) / Delta_s)
    y_non_convex=(polygon_1.y_vertices*(ss - s0) / Delta_s
              + polygon_0.y_vertices*(s1 - ss) / Delta_s)
    hull = ConvexHull(np.array([x_non_convex, y_non_convex]).T)
    i_hull = np.sort(hull.vertices)
    x_hull = x_non_convex[i_hull]
    y_hull = y_non_convex[i_hull]
    interp_polygons.append(xt.LimitPolygon(
        x_vertices=x_hull,
        y_vertices=y_hull))



t1 = time.time()
print(f'Took\t{(t1-t0)*1e3:.2f} ms')

# Visualize apertures
for ii, (trkr, poly) in enumerate(
                         zip([trk_aper_0, trk_aper_1],
                             [polygon_0, polygon_1])):
    part_gen_range = 0.05
    pp = xt.Particles(
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
plt.show()
