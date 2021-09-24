import numpy as np
import matplotlib.pyplot as plt

import xline as xl
import xtrack as xt
import xobjects as xo
import xpart as xp

plt.close('all')

n_part=10000
ctx = xo.context_default
buf = ctx.new_buffer()

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

# Aperture to polygon
i_aperture = 10

# find previous drift
import time
t0 = time.time()

ii=i_aperture
found = False
while not(found):
    ccnn = tracker.line.elements[ii].__class__.__name__
    print(ccnn)
    if ccnn == 'Drift':
        found = True
    else:
        ii -= 1
i_start = ii + 1
num_elements = i_aperture-i_start+1

# Get polygon
n_theta = 360 * 5
r_max = 0.5 # m
dr = 50e-6

theta_vect = np.linspace(0, 2*np.pi, n_theta+1)[:-1]

this_rmin = 0
this_rmax = r_max
this_dr = (this_rmax-this_rmin)/100.
rmin_theta = 0*theta_vect
for iteration in range(2):

    r_vect = np.arange(this_rmin, this_rmax, this_dr)

    RR, TT = np.meshgrid(r_vect, theta_vect)
    RR += np.atleast_2d(rmin_theta).T

    x_test = RR.flatten()*np.cos(TT.flatten())
    y_test = RR.flatten()*np.sin(TT.flatten())

    print(f'{iteration=} num_part={x_test.shape[0]}')

    ptest = xt.Particles(p0c=1,
            x = x_test.copy(),
            y = y_test.copy())
    tracker.track(ptest, ele_start=i_start, num_elements=num_elements)

    indx_sorted = np.argsort(ptest.particle_id)
    state_sorted = np.take(ptest.state, indx_sorted)

    state_mat = state_sorted.reshape(RR.shape)

    i_r_aper = np.argmin(state_mat>0, axis=1)

    rmin_theta = r_vect[i_r_aper-1]
    this_rmin=0
    this_rmax=2*this_dr
    this_dr = dr

t1 = time.time()
x_mat = x_test.reshape(RR.shape)
y_mat = y_test.reshape(RR.shape)
x_non_convex = np.array(
        [x_mat[itt, i_r_aper[itt]] for itt in range(n_theta)])
y_non_convex = np.array(
        [y_mat[itt, i_r_aper[itt]] for itt in range(n_theta)])

from scipy.spatial import ConvexHull
hull = ConvexHull(np.array([x_non_convex, y_non_convex]).T)
i_hull = np.sort(hull.vertices)
x_hull = x_non_convex[i_hull]
y_hull = y_non_convex[i_hull]

# Get a convex polygon that does not have points for all angles
temp_poly = xt.LimitPolygon(x_vertices=x_hull, y_vertices=y_hull)

# Get a convex polygon that has vertices at all requested angles
r_out = 1. # m
res = temp_poly.impact_point_and_normal(
        x_in=0*theta_vect, y_in=0*theta_vect, z_in=0*theta_vect,
        x_out=r_out*np.cos(theta_vect),
        y_out=r_out*np.sin(theta_vect),
        z_out=0*theta_vect)

polygon = xt.LimitPolygon(x_vertices=res[0], y_vertices=res[1])


t2 = time.time()
print(f'First part took\t{(t1-t0)*1e3:.2f} ms')
print(f'Second part took\t{(t2-t1)*1e3:.2f} ms')

# Visualize apertures
for ii, trkr in enumerate([trk_aper_0, trk_aper_1]):
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



plt.show()
