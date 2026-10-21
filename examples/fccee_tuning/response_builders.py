"""Analytical response builders used by the FCC-ee correction example.

These functions do not change the line. Matrices are derivatives of unweighted
residuals with respect to physical correction knobs; weights/scales belong to
Target/Vary. The approximations and signs match the original 001 script.
"""

import numpy as np


def orbit_response(reference, monitors, correctors, plane):
    """Closed-orbit displacement per positive physical kick, in the lab frame."""
    monitors = reference.rows.indices[monitors]
    correctors = reference.rows.indices[correctors]
    beta = reference['bet' + plane]
    mu = reference['mu' + plane]
    q = reference['q' + plane]
    return (
        np.sqrt(beta[monitors, None] * beta[None, correctors])
        / (2 * np.sin(np.pi * q))
        * np.cos(
            np.pi * q - 2 * np.pi * np.abs(mu[monitors, None] - mu[None, correctors])
        )
    )


def normal_quadrupole_response(
    line, reference, quadrupoles, beta_at, alpha_at, dispersion_at
):
    """Return joint log-beta/normalized-alpha/Dx/tune derivatives and knob scales.

    Columns are distributed dk1 [m^-2], one per named thick quadrupole.
    Rows: log(betx), log(bety), normalized alfx/alfy, Dx, Qx, Qy.
    Uses the four-point Gauss integration of the original FCC tuning example.
    """
    quad_indices = reference.rows.indices[quadrupoles]
    lengths = np.array([line[name].length for name in quadrupoles])
    obs_indices = reference.rows.indices[beta_at]
    alpha_indices = reference.rows.indices[alpha_at]
    dispersion_indices = reference.rows.indices[dispersion_at]

    def quad_samples(tw, plane):
        # Gauss integration of the uncoupled thick-quadrupole response.
        nodes, weights = np.polynomial.legendre.leggauss(4)
        ell = lengths[:, None] * (nodes + 1)[None, :] / 2
        weights = lengths[:, None] * weights[None, :] / 2
        sign = 1 if plane == 'x' else -1
        k1 = np.array([line[name].k1 for name in quadrupoles])
        momentum = 1 + tw.delta[quad_indices]
        k = sign * k1 / momentum
        root = np.sqrt(np.abs(k))[:, None]
        phase = root * ell
        a = np.where(k[:, None] >= 0, np.cos(phase), np.cosh(phase))
        safe = np.where(root > 0, root, 1)
        b = np.where(k[:, None] >= 0, np.sin(phase), np.sinh(phase)) / safe
        b = np.where(root > 0, b, ell) / momentum[:, None]
        beta = tw['bet' + plane][quad_indices, None]
        alpha = tw['alf' + plane][quad_indices, None]
        gamma = (1 + alpha**2) / beta
        beta_at = a * a * beta - 2 * a * b * alpha + b * b * gamma
        dmu = np.arctan2(b, a * beta - b * alpha) / (2 * np.pi)
        mu_at = tw['mu' + plane][quad_indices, None] + dmu
        dispersion_at = (
            a * tw['d' + plane][quad_indices, None]
            + b * tw['dp' + plane][quad_indices, None]
        )
        return beta_at, mu_at, weights, dispersion_at

    def beta_tune_response(tw):
        """Derivatives of log(beta_x), log(beta_y), Qx and Qy versus each dk1."""
        blocks = []
        tune_rows = []
        for plane, sign in [('x', 1), ('y', -1)]:
            beta, mu, weights, _ = quad_samples(tw, plane)
            tune = tw['q' + plane]
            response = np.zeros((len(obs_indices), len(quadrupoles)))
            for j in range(beta.shape[1]):
                diff = np.abs(tw['mu' + plane][obs_indices, None] - mu[None, :, j])
                response += (
                    -sign
                    * beta[None, :, j]
                    * weights[None, :, j]
                    * np.cos(4 * np.pi * diff - 2 * np.pi * tune)
                    / (2 * np.sin(2 * np.pi * tune))
                )
            blocks.append(response)
            tune_rows.append(sign * np.sum(beta * weights, axis=1) / (4 * np.pi))
        return np.vstack(blocks + tune_rows)

    def alpha_response(tw):
        """Derivative of alpha * beta_ref / beta - alpha_ref at nominal optics."""
        rows = []
        for plane, sign in [('x', 1), ('y', -1)]:
            beta, mu, weights, _ = quad_samples(tw, plane)
            response = np.zeros((len(alpha_indices), len(quadrupoles)))
            for j in range(beta.shape[1]):
                diff = tw['mu' + plane][alpha_indices, None] - mu[None, :, j]
                response += (
                    -sign
                    * beta[None, :, j]
                    * weights[None, :, j]
                    * np.sin(4 * np.pi * np.abs(diff) - 2 * np.pi * tw['q' + plane])
                    * np.sign(diff)
                    / (2 * np.sin(2 * np.pi * tw['q' + plane]))
                )
            rows.append(response)
        return np.vstack(rows)

    def horizontal_dispersion_response(tw):
        """Integrate -Gx(s, source)*Dx(source) through each normal quadrupole.

        A normal trim gives delta(px) = -dk1*x*ds; differentiating with respect
        to delta gives the dispersion kick -dk1*Dx*ds on the clean orbit.
        Use the same Gauss nodes and exact quadrupole transport as beta/alpha.
        """
        beta, mu, weights, dx = quad_samples(tw, 'x')
        response = np.zeros((len(dispersion_indices), len(quadrupoles)))
        for j in range(beta.shape[1]):
            diff = np.abs(tw.mux[dispersion_indices, None] - mu[None, :, j])
            green = (
                np.sqrt(tw.betx[dispersion_indices, None] * beta[None, :, j])
                * np.cos(2 * np.pi * diff - np.pi * tw.qx)
                / (2 * np.sin(np.pi * tw.qx))
            )
            response -= green * dx[None, :, j] * weights[None, :, j]
        return response

    beta_tune = beta_tune_response(reference)
    dx = horizontal_dispersion_response(reference)
    matrix = np.vstack([beta_tune[:-2], alpha_response(reference), dx, beta_tune[-2:]])
    bx, _, weights, _ = quad_samples(reference, 'x')
    by, _, _, _ = quad_samples(reference, 'y')
    scales = 1 / np.maximum(np.sum(bx * weights, axis=1), np.sum(by * weights, axis=1))
    return matrix, scales


def skew_quadrupole_response(line, reference, groups, observations):
    """Joint f1001/f1010/Dy derivative per integrated skew strength [m^-1].

    Each group is one physical sextupole, possibly split into several elements.
    Its skew strength is distributed in proportion to piece length. Responses
    use the coil midpoint. Rows: Re/Im f1001, Re/Im f1010, Dy.
    """
    skew_groups = groups
    indices = reference.rows.indices[observations]
    skew_indices = reference.rows.indices[[names[0] for names in skew_groups]]
    skew_lengths = np.array(
        [sum(line.get(n).length for n in names) for names in skew_groups]
    )
    ell = skew_lengths / 2
    skew_beta, skew_mu = {}, {}
    for plane in 'xy':
        beta = reference['bet' + plane][skew_indices]
        alpha = reference['alf' + plane][skew_indices]
        skew_beta[plane] = beta - 2 * alpha * ell + (1 + alpha**2) / beta * ell**2
        skew_mu[plane] = reference['mu' + plane][skew_indices] + np.arctan2(
            ell, beta - alpha * ell
        ) / (2 * np.pi)
    skew_scale = 1 / np.sqrt(skew_beta['x'] * skew_beta['y'])
    source_s = reference.s[skew_indices] + ell
    wrap = source_s[None, :] > reference.s[indices, None]
    dmu = {
        plane: reference['mu' + plane][indices, None]
        - skew_mu[plane][None, :]
        + wrap * reference['q' + plane]
        for plane in 'xy'
    }
    coupling_blocks = []
    for sign in (-1, 1):
        response = (
            np.sqrt(skew_beta['x'] * skew_beta['y'])[None, :]
            * np.exp(2j * np.pi * (dmu['x'] + sign * dmu['y']))
            / (4 * (1 - np.exp(2j * np.pi * (reference.qx + sign * reference.qy))))
        )
        coupling_blocks.extend([response.real, response.imag])
    # A thin skew quad gives delta(py) = delta(ksl[1])*x. Differentiating
    # with respect to delta gives a vertical kick Dx*delta(ksl[1]). In the
    # clean straight sextupole, propagate Dx to the same midpoint as beta/mu.
    skew_dx = reference.dx[skew_indices] + ell * reference.dpx[skew_indices]
    dispersion_response = (
        np.sqrt(reference.bety[indices, None] * skew_beta['y'][None, :])
        * np.cos(2 * np.pi * dmu['y'] - np.pi * reference.qy)
        / (2 * np.sin(np.pi * reference.qy))
        * skew_dx[None, :]
    )
    coupling_response = np.vstack(coupling_blocks + [dispersion_response])
    return coupling_response, skew_scale
