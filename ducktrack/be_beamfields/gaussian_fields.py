# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from scipy.constants import epsilon_0
import numpy as np


def _get_transv_field_gauss_round(sigma, Delta_x, Delta_y, x, y, mathlib):
    exp = mathlib.exp
    sqrt = mathlib.sqrt
    pi = mathlib.pi

    r2 = (x - Delta_x) * (x - Delta_x) + (y - Delta_y) * (y - Delta_y)
    if r2 < 1e-20:
        temp = sqrt(r2) / (2.0 * pi * epsilon_0 * sigma)  # linearised
    else:
        temp = (1 - exp(-0.5 * r2 / (sigma * sigma))) / (
            2.0 * pi * epsilon_0 * r2
        )

    Ex = temp * (x - Delta_x)
    Ey = temp * (y - Delta_y)

    return Ex, Ey


get_transv_field_gauss_round = np.vectorize(
    _get_transv_field_gauss_round, excluded=["mathlib"]
)


def _get_transv_field_gauss_ellip(
    sigmax, sigmay, Delta_x, Delta_y, x, y, mathlib
):

    abs = mathlib.abs
    exp = mathlib.exp
    wfun = mathlib.wfun
    sqrt = mathlib.sqrt
    pi = mathlib.pi
    sqrt_pi = sqrt(pi)

    # I always go to the first quadrant and then apply the signs a posteriori
    # numerically more stable
    # (see http://inspirehep.net/record/316705/files/slac-pub-5582.pdf)

    abx = abs(x - Delta_x)
    aby = abs(y - Delta_y)

    if sigmax > sigmay:

        S = sqrt(2.0 * (sigmax * sigmax - sigmay * sigmay))
        factBE = 1.0 / (2.0 * epsilon_0 * sqrt_pi * S)

        etaBE_re = sigmay / sigmax * abx
        etaBE_im = sigmax / sigmay * aby

        zetaBE_re = abx
        zetaBE_im = aby

        w_zetaBE_re, w_zetaBE_im = wfun(zetaBE_re / S, zetaBE_im / S)
        w_etaBE_re, w_etaBE_im = wfun(etaBE_re / S, etaBE_im / S)

        expBE = exp(
            -abx * abx / (2 * sigmax * sigmax)
            - aby * aby / (2 * sigmay * sigmay)
        )

        Ex = factBE * (w_zetaBE_im - w_etaBE_im * expBE)
        Ey = factBE * (w_zetaBE_re - w_etaBE_re * expBE)

    elif sigmay > sigmax:

        S = sqrt(2.0 * (sigmay * sigmay - sigmax * sigmax))
        factBE = 1.0 / (2.0 * epsilon_0 * sqrt_pi * S)

        etaBE_re = sigmax / sigmay * aby
        etaBE_im = sigmay / sigmax * abx

        zetaBE_re = aby
        zetaBE_im = abx

        w_zetaBE_re, w_zetaBE_im = wfun(zetaBE_re / S, zetaBE_im / S)
        w_etaBE_re, w_etaBE_im = wfun(etaBE_re / S, etaBE_im / S)

        expBE = exp(
            -aby * aby / (2 * sigmay * sigmay)
            - abx * abx / (2 * sigmax * sigmax)
        )

        Ey = factBE * (w_zetaBE_im - w_etaBE_im * expBE)
        Ex = factBE * (w_zetaBE_re - w_etaBE_re * expBE)
    else:
        Ex = 1.0 / 0.0
        Ey = 1.0 / 0.0

    if (x - Delta_x) < 0:
        Ex = -Ex
    if (y - Delta_y) < 0:
        Ey = -Ey

    return Ex, Ey


get_transv_field_gauss_ellip = np.vectorize(
    _get_transv_field_gauss_ellip, excluded=["mathlib"]
)


def _get_Ex_Ey_Gx_Gy_gauss(
    x, y, sigma_x, sigma_y, min_sigma_diff, skip_Gs, mathlib
):

    abs = mathlib.abs
    pi = mathlib.pi
    exp = mathlib.exp

    if abs(sigma_x - sigma_y) < min_sigma_diff:

        sigma = 0.5 * (sigma_x + sigma_y)
        Delta_x = 0.0
        Delta_y = 0.0

        Ex, Ey = get_transv_field_gauss_round(
            sigma, Delta_x, Delta_y, x, y, mathlib
        )

        if not skip_Gs:
            if abs(x) + abs(y) < min_sigma_diff:
                Gx = 0.0
                Gy = 0.0
            else:
                Gx = (
                    1
                    / (2.0 * (x * x + y * y))
                    * (
                        y * Ey
                        - x * Ex
                        + 1.0
                        / (2 * pi * epsilon_0 * sigma * sigma)
                        * x
                        * x
                        * exp(-(x * x + y * y) / (2.0 * sigma * sigma))
                    )
                )
                Gy = (
                    1.0
                    / (2 * (x * x + y * y))
                    * (
                        x * Ex
                        - y * Ey
                        + 1.0
                        / (2 * pi * epsilon_0 * sigma * sigma)
                        * y
                        * y
                        * exp(-(x * x + y * y) / (2.0 * sigma * sigma))
                    )
                )
    else:

        sigma_x = sigma_x
        sigma_y = sigma_y
        Delta_x = 0.0
        Delta_y = 0.0

        Ex, Ey = get_transv_field_gauss_ellip(
            sigma_x, sigma_y, Delta_x, Delta_y, x, y, mathlib
        )

        Sig_11 = sigma_x * sigma_x
        Sig_33 = sigma_y * sigma_y
        if not skip_Gs:
            Gx = (
                -1.0
                / (2 * (Sig_11 - Sig_33))
                * (
                    x * Ex
                    + y * Ey
                    + 1.0
                    / (2 * pi * epsilon_0)
                    * (
                        sigma_y
                        / sigma_x
                        * exp(-x * x / (2 * Sig_11) - y * y / (2 * Sig_33))
                        - 1.0
                    )
                )
            )
            Gy = (
                1.0
                / (2 * (Sig_11 - Sig_33))
                * (
                    x * Ex
                    + y * Ey
                    + 1.0
                    / (2 * pi * epsilon_0)
                    * (
                        sigma_x
                        / sigma_y
                        * exp(-x * x / (2 * Sig_11) - y * y / (2 * Sig_33))
                        - 1.0
                    )
                )
            )

    if skip_Gs:
        return Ex, Ey
    else:
        return Ex, Ey, Gx, Gy


get_Ex_Ey_Gx_Gy_gauss = np.vectorize(
    _get_Ex_Ey_Gx_Gy_gauss, excluded=["mathlib"]
)
