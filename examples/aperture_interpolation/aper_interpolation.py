import numpy as np
from scipy.spatial import ConvexHull

import xtrack as xt

def characterize_aperture(tracker, i_aperture, n_theta, r_max, dr,
                          buffer_for_poly):

    # find previous drift
    ii=i_aperture
    found = False
    while not(found):
        ccnn = tracker.line.elements[ii].__class__.__name__
        #print(ccnn)
        if ccnn == 'Drift':
            found = True
        else:
            ii -= 1
    i_start = ii + 1
    num_elements = i_aperture-i_start+1

    # Get polygon

    theta_vect = np.linspace(0, 2*np.pi, n_theta+1)[:-1]

    this_rmin = 0
    this_rmax = r_max
    this_dr = (this_rmax-this_rmin)/100.
    rmin_theta = 0*theta_vect
    t_iter = []
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

    x_mat = x_test.reshape(RR.shape)
    y_mat = y_test.reshape(RR.shape)
    x_non_convex = np.array(
            [x_mat[itt, i_r_aper[itt]] for itt in range(n_theta)])
    y_non_convex = np.array(
            [y_mat[itt, i_r_aper[itt]] for itt in range(n_theta)])

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

    polygon = xt.LimitPolygon(x_vertices=res[0], y_vertices=res[1],
                              _buffer=buffer_for_poly)

    return polygon, i_start

