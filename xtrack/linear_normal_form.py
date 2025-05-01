# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

from .general import _print


def healy_symplectify(M):
    # https://accelconf.web.cern.ch/e06/PAPERS/WEPCH152.PDF
    #_print("Symplectifying linear One-Turn-Map...")

    #_print("Before symplectifying: det(M) = {}".format(np.linalg.det(M)))
    I = np.identity(6)

    S = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        ]
    )

    V = np.matmul(S, np.matmul(I - M, np.linalg.inv(I + M)))
    W = (V + V.T) / 2
    if np.linalg.det(I - np.matmul(S, W)) != 0:
        M_new = np.matmul(I + np.matmul(S, W),
                          np.linalg.inv(I - np.matmul(S, W)))
    else:
        _print("WARNING: det(I - SW) = 0!")
        V_else = np.matmul(S, np.matmul(I + M, np.linalg.inv(I - M)))
        W_else = (V_else + V_else.T) / 2
        M_new = -np.matmul(
            I + np.matmul(S, W_else), np.linalg.det(I - np.matmul(S, W_else))
        )

    #_print("After symplectifying: det(M) = {}".format(np.linalg.det(M_new)))
    return M_new

S = np.array([[0., 1., 0., 0., 0., 0.],
              [-1., 0., 0., 0., 0., 0.],
              [ 0., 0., 0., 1., 0., 0.],
              [ 0., 0.,-1., 0., 0., 0.],
              [ 0., 0., 0., 0., 0., 1.],
              [ 0., 0., 0., 0.,-1., 0.]])

######################################################
### Implement Normalization of fully coupled motion ##
######################################################

def Rot2D(mu):
    return np.array([[ np.cos(mu), np.sin(mu)],
                     [-np.sin(mu), np.cos(mu)]])

def compute_linear_normal_form(M, symplectify=False, only_4d_block=False,
                        responsiveness_tol=None,
                        stability_tol=None):

    '''
    Compute the linear normal form of a 6x6 matrix M in the form:

    M = W x Rot x W^-1

    where Rot is a block diagonal matrix with 2x2 rotations.

    Parameters
    ----------
    M : np.ndarray
        6x6 matrix
    symplectify : bool
        If True, symplectify the matrix before computing the normal form
    only_4d_block : bool
        If True, only use the 4x4 block of M to compute the normal form
    responsiveness_tol : float
        Tolerance for the responsiveness of the matrix M
    stability_tol : float
        Tolerance for the stability of the matrix M

    Returns
    -------
    W : np.ndarray
        6x6 matrix
    invW: np.ndarray
        6x6 matrix
    Rot : np.ndarray
        6x6 matrix
    eigenvalues : np.ndarray
        3x1 vector
    '''

    if only_4d_block:
        M = M.copy()
        M[4:, :] = 0
        M[:, 4:] = 0
        muz_dummy = np.pi/10
        M[4:, 4:] = np.array([[np.cos(muz_dummy), np.sin(muz_dummy)],
                              [-np.sin(muz_dummy), np.cos(muz_dummy)]])

    if responsiveness_tol is not None:
        _assert_matrix_responsiveness(M, responsiveness_tol)

    if symplectify:
        M = healy_symplectify(M)

    if stability_tol is not None:
        _assert_matrix_determinant_within_tol(M, stability_tol)

    # Diagonalize M
    w0, v0 = np.linalg.eig(M)

    if stability_tol is not None:
        _assert_matrix_stability(w0, stability_tol)

    a0 = np.real(v0)
    b0 = np.imag(v0)

    index_list = [0,5,1,2,3,4] # we mix them up to check the algorithm

    ##### Sort modes in pairs of conjugate modes #####
    conj_modes = np.zeros([3,2], dtype=np.int64)
    for j in [0,1]:
        conj_modes[j,0] = index_list[0]
        del index_list[0]

        min_index = 0
        min_diff = abs(np.imag(w0[conj_modes[j,0]] + w0[index_list[min_index]]))
        for i in range(1,len(index_list)):
            diff = abs(np.imag(w0[conj_modes[j,0]] + w0[index_list[i]]))
            if min_diff > diff:
                min_diff = diff
                min_index = i

        conj_modes[j,1] = index_list[min_index]
        del index_list[min_index]

    conj_modes[2,0] = index_list[0]
    conj_modes[2,1] = index_list[1]

    ##################################################
    #### Select mode from pairs with positive (real @ S @ imag) #####

    modes = np.empty(3, dtype=np.int64)
    for ii,ind in enumerate(conj_modes):
        if np.matmul(np.matmul(a0[:,ind[0]], S), b0[:,ind[0]]) > 0:
            modes[ii] = ind[0]
        else:
            modes[ii] = ind[1]

    ##################################################
    #### Sort modes such that (1,2,3) is close to (x,y,zeta) ####
    # Identify the longitudinal mode
    for i in [0,1]:
        if abs(v0[:,modes[2]])[5] < abs(v0[:,modes[i]])[5]:
            modes[2], modes[i] = modes[i], modes[2]

    # Identify the vertical mode
    if abs(v0[:,modes[1]])[2] < abs(v0[:,modes[0]])[2]:
        modes[0], modes[1] = modes[1], modes[0]

    ##################################################
    #### Rotate eigenvectors to the Courant-Snyder parameterization ####
    phase0 = np.log(v0[0,modes[0]]).imag
    phase1 = np.log(v0[2,modes[1]]).imag
    phase2 = np.log(v0[4,modes[2]]).imag

    v0[:,modes[0]] *= np.exp(-1.j*phase0)
    v0[:,modes[1]] *= np.exp(-1.j*phase1)
    v0[:,modes[2]] *= np.exp(-1.j*phase2)

    ##################################################
    #### Construct W #################################

    a1 = v0[:,modes[0]].real
    a2 = v0[:,modes[1]].real
    a3 = v0[:,modes[2]].real
    b1 = v0[:,modes[0]].imag
    b2 = v0[:,modes[1]].imag
    b3 = v0[:,modes[2]].imag

    n1_inv_sq = np.matmul(np.matmul(a1, S), b1)
    n2_inv_sq = np.matmul(np.matmul(a2, S), b2)
    n3_inv_sq = np.matmul(np.matmul(a3, S), b3)

    if only_4d_block:
        n3_inv_sq = 1.0 # Just to avoid errors

    if not n1_inv_sq > 0:
        raise ValueError('Invalid n1')
    if not n2_inv_sq > 0:
        raise ValueError('Invalid n2')
    if not n3_inv_sq > 0 :
        raise ValueError('Invalid n3')

    n1 = 1./np.sqrt(n1_inv_sq)
    n2 = 1./np.sqrt(n2_inv_sq)
    n3 = 1./np.sqrt(n3_inv_sq)

    a1 *= n1
    a2 *= n2
    a3 *= n3

    b1 *= n1
    b2 *= n2
    b3 *= n3

    W = np.array([a1,b1,a2,b2,a3,b3]).T
    W[abs(W) < 1.e-14] = 0. # Set very small numbers to zero.
    #invW = np.matmul(np.matmul(S.T, W.T), S)
    invW = np.linalg.inv(W)

    ##################################################
    #### Get tunes and rotation matrix in the normalized coordinates ####

    mu1 = np.log(w0[modes[0]]).imag
    mu2 = np.log(w0[modes[1]]).imag
    mu3 = np.log(w0[modes[2]]).imag

    #q1 = mu1/(2.*np.pi)
    #q2 = mu2/(2.*np.pi)
    #q3 = mu3/(2.*np.pi)

    R = np.zeros_like(W)
    R[0:2,0:2] = Rot2D(mu1)
    R[2:4,2:4] = Rot2D(mu2)
    R[4:6,4:6] = Rot2D(mu3)
    ##################################################

    eigenvalues = w0[modes]

    return W, invW, R, eigenvalues

def _assert_matrix_responsiveness(M,
                responsiveness_tol, only_4d=False):
    n_check = 4 if only_4d else 6
    for ii in range(n_check):
        mask_non_zero = np.abs(M[:, ii]) > responsiveness_tol
        mask_non_zero[ii] = False
        if np.sum(mask_non_zero)<1:
            raise ValueError(
                'Invalid one-turn map: No coordinates respond to variations of '
                + 'x px y py zeta delta'.split()[ii])


def _assert_matrix_determinant_within_tol(M, tol=1e-15):
    if np.abs(np.linalg.det(M)-1) > tol:
        raise ValueError(
            f'The determinant of M is out tolerance. det={np.linalg.det(M)}')

def _assert_matrix_stability(eigenvals, tol=1e-3):
    if np.any(np.abs(eigenvals) > 1. + tol):
        raise ValueError('One-turn matrix is unstable. '
                         f'Magnitudes of eigenvalues are:\n{repr(np.abs(eigenvals))}')
