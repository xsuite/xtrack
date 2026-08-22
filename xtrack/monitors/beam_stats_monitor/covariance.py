import numpy as np


PLANES = {
    'x': ('x', 'px'),
    'y': ('y', 'py'),
    'zeta': ('zeta', 'pzeta'),
}
CANONICAL_COORDS = ('x', 'px', 'y', 'py', 'zeta', 'pzeta')
NORMAL_MODE_EMITTANCE_STATS = (
    'gemitt_x', 'gemitt_y', 'gemitt_zeta',
    'nemitt_x', 'nemitt_y', 'nemitt_zeta',
)
COVARIANCE_OPTICS_STATS = (
    'betx', 'alfx',
    'bety', 'alfy',
    'betzeta', 'alfzeta',
    'dx', 'dpx', 'dy', 'dpy',
)
COVARIANCE_DERIVED_STATS = (
    *NORMAL_MODE_EMITTANCE_STATS, *COVARIANCE_OPTICS_STATS)


def covariance_optics_from_sigma(*, sigma, num_particles, beta0_gamma0,
                                 min_num_particles=1):
    sigma = np.asarray(sigma, dtype=float)
    out = empty_covariance_optics_result(
        sigma=sigma,
        num_particles=num_particles,
        beta0_gamma0=beta0_gamma0,
        status='failed',
        message='not computed')

    if sigma.shape != (6, 6):
        out['message'] = '`sigma` must have shape (6, 6)'
        return out
    if num_particles < min_num_particles:
        out['status'] = 'insufficient_num_particles'
        out['message'] = (
            f'num_particles={num_particles} is below '
            f'min_num_particles={min_num_particles}')
        return out
    if not np.all(np.isfinite(sigma)):
        out['message'] = 'covariance matrix contains non-finite values'
        return out

    from xtrack.linear_normal_form import (
        S, sort_modes, _build_w_matrix_from_eigenvectors)

    sigma_s = sigma @ S
    try:
        out['condition_number'] = float(np.linalg.cond(sigma_s))
    except Exception:
        out['condition_number'] = np.nan

    if np.linalg.matrix_rank(sigma_s) < 6:
        out['status'] = 'rank_deficient'
        out['message'] = 'covariance matrix is rank deficient'
        return out

    try:
        eigenvalues, eigenvectors = np.linalg.eig(sigma_s)
        modes = sort_modes(eigenvectors, eigenvalues)
        w_matrix = _build_w_matrix_from_eigenvectors(eigenvectors, modes)
        from xtrack.twiss import TwissInit
        twiss_init = TwissInit(W_matrix=w_matrix)
        optics = {
            name: float(getattr(twiss_init, name))
            for name in COVARIANCE_OPTICS_STATS}
    except Exception as exc:
        out['message'] = str(exc)
        return out

    emittances = np.maximum(eigenvalues[modes].imag.real, 0.0)
    out.update({
        'status': 'ok',
        'message': '',
        'W_matrix': w_matrix,
        'gemitt_x': float(emittances[0]),
        'gemitt_y': float(emittances[1]),
        'gemitt_zeta': float(emittances[2]),
    })
    out['nemitt_x'] = out['gemitt_x'] * beta0_gamma0
    out['nemitt_y'] = out['gemitt_y'] * beta0_gamma0
    out['nemitt_zeta'] = out['gemitt_zeta'] * beta0_gamma0
    out.update(optics)
    return out


def empty_covariance_optics_result(*, sigma, num_particles, beta0_gamma0,
                                   status, message):
    out = {
        'status': status,
        'message': message,
        'covariance_matrix': np.asarray(sigma, dtype=float).copy(),
        'covariance_order': CANONICAL_COORDS,
        'W_matrix': np.full((6, 6), np.nan),
        'num_particles': float(num_particles),
        'beta0_gamma0': float(beta0_gamma0),
        'condition_number': np.nan,
    }
    for name in COVARIANCE_DERIVED_STATS:
        out[name] = np.nan
    return out
