"""Tangent map for the dipole in convergence_pkin_const.py.

Differentiate the discrete RK4 map analytically by applying the SAME stages
to z'=f(z,s) and M'=f_z(z,s) M. This is the Jacobian of RK4, not the
Jacobian of the exact flow: integration defects remain measurable, without
the cancellation noise of finite differences. See Sanz-Serna, Theorem 9
in the preprint, on RK variational equations:
https://arxiv.org/html/1503.04021#S3.SS3

This implementation is deliberately specific to a=bs=0, one normal dipole
coefficient of degree <=6, and ny=7. The calling example checks its outputs
and Jacobians against native FieldExpansion tracking. It is not a generic
replacement for Xtrack's tracking kernels.
"""

import numpy as np
from numpy.polynomial import Polynomial

import xtrack as xt


S = np.kron(np.eye(3), np.array([[0., 1.], [-1., 0.]]))


def potentials(y, derivatives):
    """Ax, dAx/dy, d2Ax/dy2 for this dipole; derivatives=(b,b',b''',b^(5))."""
    _, d1, d3, d5 = derivatives
    ax = -d1*y**2/2 + d3*y**4/24 - d5*y**6/720
    dax_dy = -d1*y + d3*y**3/6 - d5*y**5/120
    d2ax_dy2 = -d1 + d3*y**2/2 - d5*y**4/24
    return ax, dax_dy, d2ax_dy2


def rhs_and_tangent(z, matrix, derivatives, beta0):
    # Canonical coordinates: (x, px, y, py, tau=zeta/beta0, ptau).
    ax, dax_dy, d2ax_dy2 = potentials(z[:, 2], derivatives)
    u, v, e = z[:, 1] - ax, z[:, 3], z[:, 5] + 1/beta0
    r = np.sqrt(1 + 2*z[:, 5]/beta0 + z[:, 5]**2 - u**2 - v**2)
    rhs = np.zeros_like(z)
    rhs[:, 0] = u/r
    rhs[:, 1] = -derivatives[0]
    rhs[:, 2] = v/r
    rhs[:, 3] = (u/r)*dax_dy
    rhs[:, 4] = 1/beta0 - e/r

    du = np.zeros_like(z)
    du[:, 1], du[:, 2] = 1, -dax_dy
    dv, de = np.zeros_like(z), np.zeros_like(z)
    dv[:, 3], de[:, 5] = 1, 1
    dr = (e[:, None]*de - u[:, None]*du - v[:, None]*dv)/r[:, None]
    jac = np.zeros_like(matrix)
    jac[:, 0, :] = du/r[:, None] - (u/r**2)[:, None]*dr
    jac[:, 2, :] = dv/r[:, None] - (v/r**2)[:, None]*dr
    jac[:, 3, :] = dax_dy[:, None]*jac[:, 0, :]
    jac[:, 3, 2] += (u/r)*d2ax_dy2
    jac[:, 4, :] = -de/r[:, None] + (e/r**2)[:, None]*dr
    return rhs, jac @ matrix


def tangent_map(coefficients, lengths, initial, beta0, steps, pkin_const):
    """Return canonical exit coordinates and their analytic 6x6 Jacobians."""
    z = initial.copy()
    matrix = np.broadcast_to(np.eye(6), (len(z), 6, 6)).copy()
    previous_exit = None
    for coefficients_i, length in zip(coefficients, lengths):
        polynomial = Polynomial(coefficients_i)
        # Only odd derivatives enter Ax. Cache their values at RK4 nodes.
        nodes = np.linspace(0, length, 2*steps + 1)
        derivatives = np.array([polynomial.deriv(k)(nodes) for k in (0, 1, 3, 5)]).T
        if previous_exit is not None and pkin_const:
            a_old, dax_old, _ = potentials(z[:, 2], previous_exit)
            a_new, dax_new, _ = potentials(z[:, 2], derivatives[0])
            # Internal pkin-preserving interface: px += Ax_new - Ax_old.
            # Its exact Jacobian K=I+(d_y Delta Ax) e_px e_y^T is generally
            # non-symplectic, although det(K)=1. No such kick for False.
            # For C1 cubics, d_y Delta Ax = Delta b''' * y^3/6; the single
            # interface's scaled defect is L*|d_y Delta Ax|. Transported
            # contributions can cancel, so measure the full product below.
            z[:, 1] += a_new - a_old
            matrix[:, 1, :] += (dax_new - dax_old)[:, None]*matrix[:, 2, :]
        ds = length/steps
        for i in range(steps):
            k1, m1 = rhs_and_tangent(z, matrix, derivatives[2*i], beta0)
            k2, m2 = rhs_and_tangent(z + ds/2*k1, matrix + ds/2*m1,
                                      derivatives[2*i+1], beta0)
            k3, m3 = rhs_and_tangent(z + ds/2*k2, matrix + ds/2*m2,
                                      derivatives[2*i+1], beta0)
            k4, m4 = rhs_and_tangent(z + ds*k3, matrix + ds*m3,
                                      derivatives[2*i+2], beta0)
            z += ds/6*(k1 + 2*k2 + 2*k3 + k4)
            matrix += ds/6*(m1 + 2*m2 + 2*m3 + m4)
        previous_exit = derivatives[-1]
    return z, matrix


def native_map(elements, initial, particle_ref, steps, pkin_const):
    """Native tracking from entrance to exit CANONICAL coordinates."""
    particles = xt.Particles(
        _context=elements[0]._context, p0c=particle_ref.p0c[0],
        mass0=particle_ref.mass0, q0=particle_ref.q0,
        x=initial[:, 0], px=initial[:, 1], y=initial[:, 2], py=initial[:, 3],
        tau=initial[:, 4], ptau=initial[:, 5],
    )
    if pkin_const:
        entrance = elements[0].get_field(particles.x, particles.y, s=0)
        particles.px -= entrance['Ax']
        particles.py -= entrance['Ay']
    for element in elements:
        element.pkin_const = pkin_const
        element.nstep, element.ds = steps, element.length/steps
        element.track(particles)
    output = np.array([particles.x, particles.px, particles.y, particles.py,
                       particles.zeta/particles.beta0, particles.ptau]).T
    if pkin_const:
        exit_field = elements[-1].get_field(particles.x, particles.y, s=elements[-1].length)
        output[:, 1] += exit_field['Ax']
        output[:, 3] += exit_field['Ay']
    if not np.all(particles.state > 0) or not np.all(np.isfinite(output)):
        raise RuntimeError('Invalid particle in the symplecticity measurement.')
    return output


def scaled_jacobian(matrix, length):
    # Scale ALL positions by L. Each canonical pair's area has the same
    # factor 1/L, so the symplectic condition still uses the standard S.
    scales = np.array([length, 1., length, 1., length, 1.])
    return matrix*scales[None, None, :]/scales[None, :, None]


def defect(matrix):
    """Spectral norm of M^T S M-S: worst distortion of the symplectic form."""
    return np.linalg.norm(matrix.swapaxes(-1, -2) @ S @ matrix - S, ord=2, axis=(-2, -1))


def finite_difference_check(elements, initial, particle_ref, steps, mode, length, h=1e-3):
    """Fourth-order Richardson Jacobians from native tracking at h and h/2.

    Three central differences at h, h/2, h/4 give two fourth-order estimates.
    Their difference measures differentiation sensitivity, not RK4 error.
    Perturb canonical coordinates independently, keeping reference beta fixed.
    """
    scales = np.array([length, 1., length, 1., length, 1.])
    offsets = np.concatenate([sign*step*np.eye(6) for step in (h, h/2, h/4)
                              for sign in (1, -1)])
    probes = initial[:, None, :] + offsets[None, :, :]*scales
    mapped = native_map(elements, probes.reshape(-1, 6), particle_ref, steps, mode)
    mapped = (mapped/scales).reshape(len(initial), 3, 2, 6, 6)
    central = (mapped[:, :, 0] - mapped[:, :, 1]) / np.array([2*h, h, h/2])[None, :, None, None]
    central = central.swapaxes(-1, -2)  # Output row, input column.
    coarse = (4*central[:, 1] - central[:, 0])/3
    fine = (4*central[:, 2] - central[:, 1])/3
    sensitivity = np.linalg.norm(fine - coarse, ord=2, axis=(-2, -1))
    return fine, sensitivity
