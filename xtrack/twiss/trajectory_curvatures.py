# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np


def _get_trajectory_curvatures(twiss_res):

    angle = twiss_res['angle']
    rot_s_rad = twiss_res['rot_s_rad']
    x = twiss_res['x']
    y = twiss_res['y']
    kin_px = twiss_res['kin_px']
    kin_py = twiss_res['kin_py']
    delta = twiss_res['delta']
    length = twiss_res['length']

    # Curvature of the reference trajectory
    mask = length != 0
    kappa0_x = 0 * angle
    kappa0_y = 0 * angle
    kappa0_x[mask] = angle[mask] * np.cos(rot_s_rad[mask]) / length[mask]
    kappa0_y[mask] = angle[mask] * np.sin(rot_s_rad[mask]) / length[mask]

    # Compute x', y', x'', y''
    ps = np.sqrt((1 + delta)**2 - kin_px**2 - kin_py**2)
    xp = kin_px / ps
    yp = kin_py / ps
    xp_ele = xp * 0
    yp_ele = yp * 0
    xp_ele[:-1] = (xp[:-1] + xp[1:]) / 2
    yp_ele[:-1] = (yp[:-1] + yp[1:]) / 2

    mask_length = length != 0
    xpp_ele = xp_ele * 0
    ypp_ele = yp_ele * 0
    xpp_ele[mask_length] = np.diff(xp, append=0)[mask_length] / length[mask_length]
    ypp_ele[mask_length] = np.diff(yp, append=0)[mask_length] / length[mask_length]

    # Curvature of the particle trajectory
    hhh = 1 + kappa0_x * x + kappa0_y * y
    hprime = kappa0_x * xp_ele + kappa0_y * yp_ele
    mask = hhh**2 != 0
    kappa_x = (-(hhh * (xpp_ele - hhh * kappa0_x) - 2 * hprime * xp_ele)[mask]
            / (xp_ele**2 + hhh**2)[mask]**(3/2))
    kappa_y = (-(hhh * (ypp_ele - hhh * kappa0_y) - 2 * hprime * yp_ele)[mask]
            / (yp_ele**2 + hhh**2)[mask]**(3/2))

    return kappa_x, kappa_y, kappa0_x, kappa0_y
