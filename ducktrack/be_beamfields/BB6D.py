# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from . import propagate_sigma_matrix as psm
from . import boost as boost

# import transverse_efields as tef

from . import gaussian_fields as gf


import numpy as np


from scipy.constants import c as c_light


def BB6D_track(x, px, y, py, sigma, delta, q0, p0, bb6ddata, mathlib):

    # print('Input px',px)

    q_part = bb6ddata.q_part
    parboost = bb6ddata.parboost
    Sigmas_0_star = bb6ddata.Sigmas_0_star
    N_slices = bb6ddata.N_slices
    N_part_per_slice = bb6ddata.N_part_per_slice
    x_slices_star = bb6ddata.x_slices_star
    y_slices_star = bb6ddata.y_slices_star
    sigma_slices_star = bb6ddata.sigma_slices_star
    min_sigma_diff = bb6ddata.min_sigma_diff
    threshold_singular = bb6ddata.threshold_singular

    # Change reference frame
    x_subCO = x - bb6ddata.x_CO - bb6ddata.delta_x
    px_subCO = px - bb6ddata.px_CO
    y_subCO = y - bb6ddata.y_CO - bb6ddata.delta_y
    py_subCO = py - bb6ddata.py_CO
    sigma_subCO = sigma - bb6ddata.sigma_CO
    delta_subCO = delta - bb6ddata.delta_CO

    # Boost coordinates of the weak beam
    x_star, px_star, y_star, py_star, sigma_star, delta_star = boost.boost(
        x_subCO,
        px_subCO,
        y_subCO,
        py_subCO,
        sigma_subCO,
        delta_subCO,
        parboost,
    )
    # ~ x_star, px_star, y_star, py_star, sigma_star, delta_star =
    #      (x, px, y, py, sigma, delta)
    for i_slice in range(N_slices):
        sigma_slice_star = sigma_slices_star[i_slice]
        x_slice_star = x_slices_star[i_slice]
        y_slice_star = y_slices_star[i_slice]

        # Compute force scaling factor
        Ksl = N_part_per_slice[i_slice] * q_part * q0 / (p0 * c_light)

        # Identify the Collision Point (CP)
        S = 0.5 * (sigma_star - sigma_slice_star)

        # Get strong beam shape at the CP
        (
            Sig_11_hat_star,
            Sig_33_hat_star,
            costheta,
            sintheta,
            dS_Sig_11_hat_star,
            dS_Sig_33_hat_star,
            dS_costheta,
            dS_sintheta,
            extra_data,
        ) = psm.propagate_Sigma_matrix(
            Sigmas_0_star, S, threshold_singular=threshold_singular
        )

        # Evaluate transverse coordinates of the weake baem w.r.t.
        # the strong beam centroid
        x_bar_star = x_star + px_star * S - x_slice_star
        y_bar_star = y_star + py_star * S - y_slice_star

        # Move to the uncoupled reference frame
        x_bar_hat_star = x_bar_star * costheta + y_bar_star * sintheta
        y_bar_hat_star = -x_bar_star * sintheta + y_bar_star * costheta

        # Compute derivatives of the transformation
        dS_x_bar_hat_star = x_bar_star * dS_costheta + y_bar_star * dS_sintheta
        dS_y_bar_hat_star = (
            -x_bar_star * dS_sintheta + y_bar_star * dS_costheta
        )

        # Compute normalized field
        # Ex, Ey, Gx, Gy = tef.get_Ex_Ey_Gx_Gy_gauss(
        #                     x=x_bar_hat_star, y=y_bar_hat_star,
        #                     sigma_x=np.sqrt(Sig_11_hat_star),
        #                     sigma_y=np.sqrt(Sig_33_hat_star),
        #                     min_sigma_diff = min_sigma_diff)

        Ex, Ey, Gx, Gy = gf.get_Ex_Ey_Gx_Gy_gauss(
            x=x_bar_hat_star,
            y=y_bar_hat_star,
            sigma_x=np.sqrt(Sig_11_hat_star),
            sigma_y=np.sqrt(Sig_33_hat_star),
            min_sigma_diff=min_sigma_diff,
            skip_Gs=False,
            mathlib=mathlib,
        )

        # Compute kicks
        Fx_hat_star = Ksl * Ex
        Fy_hat_star = Ksl * Ey
        Gx_hat_star = Ksl * Gx
        Gy_hat_star = Ksl * Gy

        # Move kisks to coupled reference frame
        Fx_star = Fx_hat_star * costheta - Fy_hat_star * sintheta
        Fy_star = Fx_hat_star * sintheta + Fy_hat_star * costheta

        # Compute longitudinal kick
        Fz_star = 0.5 * (
            Fx_hat_star * dS_x_bar_hat_star
            + Fy_hat_star * dS_y_bar_hat_star
            + Gx_hat_star * dS_Sig_11_hat_star
            + Gy_hat_star * dS_Sig_33_hat_star
        )

        # Apply the kicks (Hirata's synchro-beam)
        delta_star = (
            delta_star
            + Fz_star
            + 0.5
            * (
                Fx_star * (px_star + 0.5 * Fx_star)
                + Fy_star * (py_star + 0.5 * Fy_star)
            )
        )
        x_star = x_star - S * Fx_star
        px_star = px_star + Fx_star
        y_star = y_star - S * Fy_star
        py_star = py_star + Fy_star

    # Inverse boost on the coordinates of the weak beam
    x_ret, px_ret, y_ret, py_ret, sigma_ret, delta_ret = boost.inv_boost(
        x_star, px_star, y_star, py_star, sigma_star, delta_star, parboost
    )

    # Go back to original reference frame and remove dipolar effect
    x_out = x_ret + bb6ddata.x_CO + bb6ddata.delta_x - bb6ddata.Dx_sub
    px_out = px_ret + bb6ddata.px_CO - bb6ddata.Dpx_sub
    y_out = y_ret + bb6ddata.y_CO + bb6ddata.delta_y - bb6ddata.Dy_sub
    py_out = py_ret + bb6ddata.py_CO - bb6ddata.Dpy_sub
    sigma_out = sigma_ret + bb6ddata.sigma_CO - bb6ddata.Dsigma_sub
    delta_out = delta_ret + bb6ddata.delta_CO - bb6ddata.Ddelta_sub

    # print(x_ret, px_ret, y_ret, py_ret, sigma_ret, delta_ret)
    return x_out, px_out, y_out, py_out, sigma_out, delta_out
