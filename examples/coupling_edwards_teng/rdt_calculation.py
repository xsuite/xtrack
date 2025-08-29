import numpy as np
def compute_rdt(r11, r12, r21, r22, betx, bety, alfx, alfy):

    '''
    Developed by CERN OMC team.
    Ported from:
    https://pypi.org/project/optics-functions/
    https://github.com/pylhc/optics_functions
    '''

    n = len(r11)
    assert len(r12) == n
    assert len(r21) == n
    assert len(r22) == n
    gx, r, inv_gy = np.zeros((n, 2, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2, 2))

    # Eq. (16)  C = 1 / (1 + |R|) * -J R J
    # rs form after -J R^T J
    r[:, 0, 0] = r22
    r[:, 0, 1] = -r12
    r[:, 1, 0] = -r21
    r[:, 1, 1] = r11

    r *= 1 / np.sqrt(1 + np.linalg.det(r)[:, None, None])

    # Cbar = Gx * C * Gy^-1,   Eq. (5)
    sqrt_betax = np.sqrt(betx)
    sqrt_betay = np.sqrt(bety)

    gx[:, 0, 0] = 1 / sqrt_betax
    gx[:, 1, 0] = alfx * gx[:, 0, 0]
    gx[:, 1, 1] = sqrt_betax

    inv_gy[:, 1, 1] = 1 / sqrt_betay
    inv_gy[:, 1, 0] = -alfy * inv_gy[:, 1, 1]
    inv_gy[:, 0, 0] = sqrt_betay

    c = np.matmul(gx, np.matmul(r, inv_gy))
    gamma = np.sqrt(1 - np.linalg.det(c))

    # Eq. (9) and Eq. (10)
    denom = 1 / (4 * gamma)
    f1001 = denom * (+c[:, 0, 1] - c[:, 1, 0] + (c[:, 0, 0] + c[:, 1, 1]) * 1j)
    f1010 = denom * (-c[:, 0, 1] - c[:, 1, 0] + (c[:, 0, 0] - c[:, 1, 1]) * 1j)

    return {'f1001': f1001, 'f1010': f1010}
