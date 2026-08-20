# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

from .. import linear_normal_form as lnf


def _get_coupling_elements_edwards_teng(
        W_matrix: np.ndarray,
        qx: float,
        qy: float,
):
    """Compute the Edwards-Teng coupling matrix at all locations."""

    R_edw_teng = _edwards_teng_from_one_turn_at_all_locations(
        W_matrix, qx, qy)

    return {
        'r11_edw_teng': R_edw_teng[:, 0, 0],
        'r12_edw_teng': R_edw_teng[:, 0, 1],
        'r21_edw_teng': R_edw_teng[:, 1, 0],
        'r22_edw_teng': R_edw_teng[:, 1, 1],
    }


def _get_edwards_teng_matrix(R):

    A = R[:2, :2]
    B = R[:2, 2:4]
    C = R[2:4, :2]
    D = R[2:4, 2:4]

    if np.linalg.norm(B) < 1e-10 and np.linalg.norm(C) < 1e-10:
        return np.zeros((2, 2))

    C_plus_B_bar = C + _conj_mat(B)
    det_C_plus_B_bar = np.linalg.det(C_plus_B_bar)
    trace_A_minus_trace_D = np.trace(A) - np.trace(D)
    denominator = -(
        0.5 * trace_A_minus_trace_D
        + np.sign(trace_A_minus_trace_D) * np.sqrt(
            det_C_plus_B_bar + 0.25 * trace_A_minus_trace_D**2)
    )

    return C_plus_B_bar / denominator


def _conj_mat(matrix):
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[1, 0]
    d = matrix[1, 1]
    return np.array([[d, -b], [-c, a]])


def _edwards_teng_from_one_turn_at_all_locations(W_matrix, qx, qy):

    rotation = np.zeros(shape=(6, 6), dtype=np.float64)
    rotation[0:2, 0:2] = lnf.Rot2D(2 * np.pi * qx)
    rotation[2:4, 2:4] = lnf.Rot2D(2 * np.pi * qy)

    R_edw_teng = np.zeros((W_matrix.shape[0], 2, 2))

    for ii, W_at_element in enumerate(W_matrix):
        W_at_element_inv = lnf.S.T @ W_at_element.T @ lnf.S
        R_one_turn = W_at_element @ rotation @ W_at_element_inv
        R_edw_teng[ii] = _get_edwards_teng_matrix(R_one_turn)

    return R_edw_teng
