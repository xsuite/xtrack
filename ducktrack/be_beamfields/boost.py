# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

# I program as close as possible to C...


def boost(x, px, y, py, sigma, delta, parboost):

    sphi = parboost.sphi
    cphi = parboost.cphi
    tphi = parboost.tphi
    salpha = parboost.salpha
    calpha = parboost.calpha

    h = (
        delta
        + 1.0
        - np.sqrt((1.0 + delta) * (1.0 + delta) - px * px - py * py)
    )

    px_st = px / cphi - h * calpha * tphi / cphi
    py_st = py / cphi - h * salpha * tphi / cphi
    delta_st = (
        delta - px * calpha * tphi - py * salpha * tphi + h * tphi * tphi
    )

    pz_st = np.sqrt(
        (1.0 + delta_st) * (1.0 + delta_st) - px_st * px_st - py_st * py_st
    )
    hx_st = px_st / pz_st
    hy_st = py_st / pz_st
    hsigma_st = 1.0 - (delta_st + 1) / pz_st

    L11 = 1.0 + hx_st * calpha * sphi
    L12 = hx_st * salpha * sphi
    L13 = calpha * tphi

    L21 = hy_st * calpha * sphi
    L22 = 1.0 + hy_st * salpha * sphi
    L23 = salpha * tphi

    L31 = hsigma_st * calpha * sphi
    L32 = hsigma_st * salpha * sphi
    L33 = 1.0 / cphi

    x_st = L11 * x + L12 * y + L13 * sigma
    y_st = L21 * x + L22 * y + L23 * sigma
    sigma_st = L31 * x + L32 * y + L33 * sigma

    return x_st, px_st, y_st, py_st, sigma_st, delta_st


def inv_boost(x_st, px_st, y_st, py_st, sigma_st, delta_st, parboost):

    sphi = parboost.sphi
    cphi = parboost.cphi
    tphi = parboost.tphi
    salpha = parboost.salpha
    calpha = parboost.calpha

    pz_st = np.sqrt(
        (1.0 + delta_st) * (1.0 + delta_st) - px_st * px_st - py_st * py_st
    )
    hx_st = px_st / pz_st
    hy_st = py_st / pz_st
    hsigma_st = 1.0 - (delta_st + 1) / pz_st

    Det_L = (
        1.0 / cphi
        + (hx_st * calpha + hy_st * salpha - hsigma_st * sphi) * tphi
    )

    Linv_11 = (
        1.0 / cphi + salpha * tphi * (hy_st - hsigma_st * salpha * sphi)
    ) / Det_L
    Linv_12 = (salpha * tphi * (hsigma_st * calpha * sphi - hx_st)) / Det_L
    Linv_13 = (
        -tphi
        * (
            calpha
            - hx_st * salpha * salpha * sphi
            + hy_st * calpha * salpha * sphi
        )
        / Det_L
    )

    Linv_21 = (calpha * tphi * (-hy_st + hsigma_st * salpha * sphi)) / Det_L
    Linv_22 = (
        1.0 / cphi + calpha * tphi * (hx_st - hsigma_st * calpha * sphi)
    ) / Det_L
    Linv_23 = (
        -tphi
        * (
            salpha
            - hy_st * calpha * calpha * sphi
            + hx_st * calpha * salpha * sphi
        )
        / Det_L
    )

    Linv_31 = -hsigma_st * calpha * sphi / Det_L
    Linv_32 = -hsigma_st * salpha * sphi / Det_L
    Linv_33 = (1.0 + hx_st * calpha * sphi + hy_st * salpha * sphi) / Det_L

    x_i = Linv_11 * x_st + Linv_12 * y_st + Linv_13 * sigma_st
    y_i = Linv_21 * x_st + Linv_22 * y_st + Linv_23 * sigma_st
    sigma_i = Linv_31 * x_st + Linv_32 * y_st + Linv_33 * sigma_st

    h = (delta_st + 1.0 - pz_st) * cphi * cphi

    px_i = px_st * cphi + h * calpha * tphi
    py_i = py_st * cphi + h * salpha * tphi

    delta_i = (
        delta_st
        + px_i * calpha * tphi
        + py_i * salpha * tphi
        - h * tphi * tphi
    )

    return x_i, px_i, y_i, py_i, sigma_i, delta_i
