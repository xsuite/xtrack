# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import hbar

import xtrack as xt  # To avoid circular imports


_SPIN_EIGENVALUE_TOL = 1e-3
_SPIN_ROTATION_IDENTITY_TOL = 1e-13
_SPIN_AXIS_SIGN_TOL = 1e-12


def _compute_spin_rotation_matrix(line, particle_on_co):
    # Track each basis spin on an exact copy of the closed-orbit particle.
    # In kick_as_co mode the first particle's energy update and the copied
    # kicks differ slightly in roundoff; separate tracks keep the orbit
    # identical for all three columns, including at each radiation kick.
    rotation = np.empty((3, 3))
    for column, component in enumerate(['spin_x', 'spin_y', 'spin_z']):
        particle = particle_on_co.copy()
        particle.spin_x = 0.
        particle.spin_y = 0.
        particle.spin_z = 0.
        setattr(particle, component, 1.)
        line.track(particle)
        if particle.state[0] <= 0:
            raise ValueError('Particle lost while computing the spin rotation matrix')
        rotation[:, column] = [particle.spin_x[0], particle.spin_y[0],
                               particle.spin_z[0]]
    return rotation


def _find_spin_fixed_point(line, particle_on_co):

    with xt.line._preserve_config(line):
        # Spin is behind the same compile flag as synchrotron radiation.
        line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
        rotation = _compute_spin_rotation_matrix(line, particle_on_co)
        if np.max(np.abs(rotation - np.eye(3))) < _SPIN_ROTATION_IDENTITY_TOL:
            # The identity has no distinguished axis. Retain a supplied spin
            # direction, or use the vertical direction when none is supplied.
            n0 = np.array([particle_on_co.spin_x[0], particle_on_co.spin_y[0],
                           particle_on_co.spin_z[0]])
            # Check for exact zero: it means no initial spin direction was
            # supplied. Any nonzero vector provides an axis to normalize.
            if np.linalg.norm(n0) == 0:
                n0 = np.array([0., 1., 0.])
        else:
            # The invariant spin direction n0 is the eigenvector of the
            # one-turn spin rotation matrix with eigenvalue one: R @ n0 = n0.
            eigenvalues, eigenvectors = np.linalg.eig(rotation)
            i_fixed = np.argmin(np.abs(eigenvalues - 1))
            if abs(eigenvalues[i_fixed] - 1) > _SPIN_EIGENVALUE_TOL:
                raise ValueError(
                    'Spin rotation matrix has no eigenvalue sufficiently close '
                    f'to one (closest: {eigenvalues[i_fixed]}, '
                    f'tolerance: {_SPIN_EIGENVALUE_TOL})')
            n0 = eigenvectors[:, i_fixed].real
        n0 /= np.linalg.norm(n0)

        # Keep the usual positive vertical orientation. For a horizontal axis,
        # use its largest component to choose a deterministic sign.
        sign_component = (1 if abs(n0[1]) > _SPIN_AXIS_SIGN_TOL
                          else np.argmax(np.abs(n0)))
        if n0[sign_component] < 0:
            n0 *= -1

    return tuple(n0)


def _add_spin_polarization(tw, line, method):

    with xt.line._preserve_config(line):

        line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False # For spin

        # Based on:
        # A. Chao, Evaluation of Radiative Spin Polarization in an Electron Storage Ring
        # https://inspirehep.net/literature/154360

        steps_R_matrix = tw.steps_R_matrix

        for kk in steps_R_matrix:
            steps_R_matrix[kk] *= 0.1

        out = line.get_R_matrix(particle_on_co=tw.particle_on_co,
                                                            element_by_element=True,
                                                            steps=steps_R_matrix)
        mon_r_ebe = out['mon_ebe']
        part = out['part_temp']

        steps_R_matrix = out['steps_R_matrix']

        dx = steps_R_matrix["dx"]
        dpx = steps_R_matrix["dpx"]
        dy = steps_R_matrix["dy"]
        dpy = steps_R_matrix["dpy"]
        dzeta = steps_R_matrix["dzeta"]
        ddelta = steps_R_matrix["ddelta"]

        # Use the initial perturbation, as in the orbital response matrix.
        # The energy separation can change during the turn (e.g. with RF).
        dpzeta = float(mon_r_ebe.pzeta[6, 0] - mon_r_ebe.pzeta[12, 0]) / 2

        temp_mat = np.zeros((3, len(part.spin_x)))
        temp_mat[0, :] = part.spin_x
        temp_mat[1, :] = part.spin_y
        temp_mat[2, :] = part.spin_z

        DD = np.zeros((3, 6))

        for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, dpzeta]):
            DD[:, jj] = (temp_mat[:, jj+1] - temp_mat[:, jj+1+6])/(2*dd)

        RR = np.eye(9)
        RR_orb = out['R_matrix'].copy()
        RR[:6, :6] = out['R_matrix']
        RR[6:, :6] = DD

        # Spin response matrix
        ds = 1e-5

        import xpart as xp
        A = _compute_spin_rotation_matrix(line, tw.particle_on_co)

        RR[6:, 6:] = A

        # For the spin tune I take the eigenvalue with the largest imaginary part
        # (there are three eigenvalues, one is 1.0 + 0j, the others are complex conjugates)
        spin_tune_fractional = np.max(np.angle(np.linalg.eigvals(A))) / (2 * np.pi)

        # Detect no RF
        if np.abs(RR[5, 4]) < 1e-12:
            assert method == '4d'

        if method == '4d':
            RR_for_eig = np.delete(np.delete(RR, 4, axis=0), 4, axis=1)
        else:
            RR_for_eig = RR

        eival_all, eivec_all = np.linalg.eig(RR_for_eig)

        # Suppress the 4th row and col
        if method == '4d':
            RR_orb = np.delete(RR_orb, 4, axis=0)
            RR_orb = np.delete(RR_orb, 4, axis=1)

        eival, EE_orb = np.linalg.eig(RR_orb)
        n_eigen = EE_orb.shape[1]

        # Add a dummy row 4 in eivec
        if method == '4d':
            EE_orb = np.insert(EE_orb, 4, 0, axis=0)

        # A derivative of a unit spin is perpendicular to n0. Solve in that
        # two-dimensional plane: in 4D the energy mode has eigenvalue one,
        # making the unconstrained 3D system singular along n0.
        n0 = np.array([tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]])
        _, _, vh = np.linalg.svd(n0[None, :], full_matrices=True)
        spin_basis = vh[1:, :].T
        A_perp = spin_basis.T @ A @ spin_basis
        DD_perp = spin_basis.T @ DD
        EE_spin = np.zeros((3, n_eigen), dtype=complex)
        for ii in range(n_eigen):
            EE_spin[:, ii] = spin_basis @ np.linalg.solve(
                eival[ii] * np.eye(2) - A_perp, DD_perp @ EE_orb[:, ii])

        eee = np.zeros((9, n_eigen), dtype=complex)
        eee[:6, :] = EE_orb
        eee[6:, :] = EE_spin

        # Scale and track eigenvectors
        def get_scale(e):
            return np.max([np.abs(e[0])/dx, np.abs(e[1])/dpx,
                        np.abs(e[2])/dy, np.abs(e[3])/dpy,
                        np.abs(e[4])/dzeta, np.abs(e[5])/dpzeta,
                        np.abs(e[6])/ds, np.abs(e[7])/ds,
                        np.abs(e[8])/ds,
                        ])

        scales = [get_scale(eee[:, ii]) for ii in range(n_eigen)]

        eee_scaled = np.zeros((9, n_eigen), dtype=complex)
        for ii in range(n_eigen):
            eee_scaled[:, ii] = eee[:, ii] / scales[ii]

        EE_side = {}

        for side in [1, -1]:

            eee_trk_re = side * eee_scaled.real
            eee_trk_im = side * eee_scaled.imag

            particle_data = {}
            # Radiation modes that reuse the first particle's kicks or scaling
            # require the closed-orbit particle at the head of the batch.
            for ii, key in enumerate(['x', 'px', 'y', 'py', 'zeta', 'pzeta',
                                    'spin_x', 'spin_y', 'spin_z']):
                particle_data[key] = tw[key][0] + np.array(
                    [0.] + list(eee_trk_re[ii, :]) + list(eee_trk_im[ii, :])
                )

            par_track = xp.build_particles(
                particle_ref=tw.particle_on_co, mode='set', **particle_data
            )

            line.track(par_track, turn_by_turn_monitor='ONE_TURN_EBE')
            mon_ebe = line.record_last_track

            ee_ebe = np.zeros((len(tw), 9, n_eigen), dtype=complex)

            for ii, key in enumerate(['x', 'px', 'y', 'py', 'zeta', 'pzeta',
                                    'spin_x', 'spin_y', 'spin_z']):
                mon_vv = getattr(mon_ebe, key)
                for iee in range(n_eigen):
                    ee_ebe[:, ii, iee] = side *((mon_vv[1 + iee, :] - tw[key])
                                    + 1j * (mon_vv[1 + n_eigen + iee, :] - tw[key]))

            # Rephase
            for ii in range(n_eigen):
                i_max = np.argmax(np.abs(ee_ebe[0, :, ii])) # Strongest component at the start of the ring
                this_phi = np.angle(ee_ebe[:, i_max, ii])
                for jj in range(ee_ebe.shape[1]):
                    ee_ebe[:, jj, ii] *= np.exp(-1j * this_phi)

            EE = ee_ebe.copy()

            EE_side[side] = EE

        # Average the two sides
        EE = 0.5 * (EE_side[1] + EE_side[-1])
        EE_orb  = EE[:, :6, :]
        EE_spin = EE[:, 6:, :]

        if method == '4d':
            # Remove the 4th row
            EE_orb = np.delete(EE_orb, 4, axis=1)

        # In the future we could add a filter to select certain modes
        # fltr = np.diag([1, 1, 1, 1, 1]) # to select only certain modes
        fltr = np.eye(EE_orb.shape[1]) # for now

        NN = np.real(EE_spin @ fltr @ np.linalg.inv(EE_orb))
        # NN describes n(u) = n0 + NN @ u to first order in orbital offsets u.
        # Differentiating n(u) . n(u) = 1 gives n0.T @ NN = 0: every derivative
        # column must be perpendicular to n0. At each lattice location, apply
        # NN <- (I - n0 @ n0.T) @ NN, removing each column's component along n0.
        # This numerical cleanup assumes that the remaining parallel component
        # comes from finite-difference error.
        n0_ebe = np.array([tw.spin_x, tw.spin_y, tw.spin_z]).T
        n0_ebe /= np.linalg.norm(n0_ebe, axis=1)[:, None]
        NN -= n0_ebe[:, :, None] * (n0_ebe[:, None, :] @ NN)
        if method == '4d':
            # Add a dummy col 4 in NN
            NN = np.insert(NN, 4, 0, axis=2)
        # d(pzeta)/d(delta) = beta / beta0, including off-momentum orbits.
        rvv = (1 + tw.delta) / (1 + tw.beta0 * tw.ptau)
        dn_ddelta = NN[:, :, 5] * rvv[:, None]

        dn_ddelta_mod = np.sqrt(dn_ddelta[:, 0]**2
                                    + dn_ddelta[:, 1]**2
                                    + dn_ddelta[:, 2]**2)

        kappa_x = tw.rad_int_kappa_x
        kappa_y = tw.rad_int_kappa_y
        kappa = tw.rad_int_kappa
        iv_x = tw.rad_int_iv_x
        iv_y = tw.rad_int_iv_y
        iv_z = tw.rad_int_iv_z

        n0_iv = tw.spin_x * iv_x + tw.spin_y * iv_y + tw.spin_z * iv_z
        r0 = tw.particle_on_co.get_classical_particle_radius0()
        m0_J = tw.particle_on_co.mass0 * qe
        m0_kg = m0_J / clight**2

        # reference https://lib-extopc.kek.jp/preprints/PDF/1980/8011/8011060.pdf
        brho_ref = tw.particle_on_co.p0c[0] / clight / tw.particle_on_co.q0
        brho_part = (brho_ref * tw.particle_on_co.rvv[0] * tw.particle_on_co.energy[0]
                    / tw.particle_on_co.energy0[0])

        By = kappa_x * brho_part
        Bx = -kappa_y * brho_part
        Bz = tw.ks * brho_ref + tw.bs
        B_mod = np.sqrt(Bx**2 + By**2 + Bz**2)
        B_mod[B_mod == 0] = 999. # avoid division by zero

        ib_x = Bx / B_mod
        ib_y = By / B_mod
        ib_z = Bz / B_mod

        n0_ib = tw.spin_x * ib_x + tw.spin_y * ib_y + tw.spin_z * ib_z
        dn_ddelta_ib = (dn_ddelta[:, 0] * ib_x
                            + dn_ddelta[:, 1] * ib_y
                            + dn_ddelta[:, 2] * ib_z)

        int_kappa3_n0_ib = np.sum(kappa**3 * n0_ib * tw.length)
        int_kappa3_dn_ddelta_ib = np.sum(kappa**3 * dn_ddelta_ib * tw.length)
        int_kappa3_11_18_dn_ddelta_sq = 11./18. * np.sum(kappa**3 * dn_ddelta_mod**2 * tw.length)

        alpha_minus_co = 1. / tw.line_length * np.sum(kappa**3 * n0_ib *  tw.length)

        alpha_plus_co = 1. / tw.line_length * np.sum(
            kappa**3 * (1 - 2./9. * n0_iv**2) * tw.length)

        alpha_plus = alpha_plus_co + int_kappa3_11_18_dn_ddelta_sq / tw.line_length
        alpha_minus = alpha_minus_co - int_kappa3_dn_ddelta_ib / tw.line_length

        pol_inf = 8 / 5 / np.sqrt(3) * alpha_minus_co / alpha_plus_co
        pol_eq = 8 / 5 / np.sqrt(3) * alpha_minus / alpha_plus

        one_over_t_pol_component_s = (
            5 * np.sqrt(3) / 8 * r0 * hbar * tw.gamma0**5 / m0_kg * alpha_plus_co)
        one_over_t_pol_buildup_s = (
            5 * np.sqrt(3) / 8 * r0 * hbar * tw.gamma0**5 / m0_kg * alpha_plus)

        one_over_t_depol_component_s = one_over_t_pol_buildup_s - one_over_t_pol_component_s

        t_pol_component_s = 1 / one_over_t_pol_component_s
        t_pol_buildup_s = 1 / one_over_t_pol_buildup_s
        # A zero depolarization rate corresponds to an infinite lifetime,
        # as in the planar limit. Handle it explicitly to avoid dividing by zero.
        t_depol_component_s = (np.inf if one_over_t_depol_component_s == 0
                               else 1 / one_over_t_depol_component_s)

        cols = {
            'spin_dn_ddelta_x': dn_ddelta[:, 0],
            'spin_dn_ddelta_y': dn_ddelta[:, 1],
            'spin_dn_ddelta_z': dn_ddelta[:, 2],
            'spin_eigenvectors': EE,
            'spin_n_matrix': NN,
            'spin_n0_iv': n0_iv,
            'spin_n0_ib': n0_ib,
        }

        other_data = {
            'spin_tune_fractional': spin_tune_fractional,
            'spin_polarization_eq': pol_eq,
            'spin_t_pol_buildup_s': t_pol_buildup_s,
            'spin_polarization_inf_no_depol': pol_inf,
            'spin_alpha_plus_co': alpha_plus_co,
            'spin_alpha_minus_co': alpha_minus_co,
            'spin_alpha_plus': alpha_plus,
            'spin_alpha_minus': alpha_minus,
            'spin_int_kappa3_n0_ib': int_kappa3_n0_ib,
            'spin_int_kappa3_dn_ddelta_ib': int_kappa3_dn_ddelta_ib,
            'spin_int_kappa3_11_18_dn_ddelta_sq': int_kappa3_11_18_dn_ddelta_sq,
            'spin_t_pol_component_s': t_pol_component_s,
            'spin_t_depol_component_s': t_depol_component_s,

            # For diagnostics
            '_spin_ee_side': EE_side,
            '_spin_scale_factors': scales,
            '_spin_eee_trk_re': eee_trk_re,
            '_spin_eee_trk_im': eee_trk_im,
        }

        for nn in cols:
            tw[nn] = cols[nn]

        for nn in other_data:
            tw._data[nn] = other_data[nn]
