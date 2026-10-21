import numpy as np

COORDS = ['x', 'px', 'y', 'py', 'zeta', 'pzeta']
STEPS = np.array([1e-7, 1e-8, 1e-7, 1e-8, 1e-7, 1e-8])


def fd_jacobian(line, centre):
    """Jacobian of one pass through ``line`` at ``centre`` (x, px, y, py,
    zeta, pzeta), by central differences, one coordinate at a time."""
    start = np.repeat(np.asarray(centre, dtype=float)[:, None], 12, axis=1)
    for ii in range(6):
        start[ii, 2 * ii] += STEPS[ii]
        start[ii, 2 * ii + 1] -= STEPS[ii]
    p = line.build_particles(**dict(zip(COORDS, start)))
    line.track(p)
    end = np.array([getattr(p, cc) for cc in COORDS])
    jac = np.zeros((6, 6))
    for ii in range(6):
        jac[:, ii] = (end[:, 2 * ii] - end[:, 2 * ii + 1]) / (2 * STEPS[ii])
    return jac


def assert_symplectic(jac, atol=1e-6):
    S = np.zeros((6, 6))
    for ii in range(3):
        S[2 * ii, 2 * ii + 1] = 1
        S[2 * ii + 1, 2 * ii] = -1
    np.testing.assert_allclose(jac.T @ S @ jac, S, rtol=0, atol=atol)
