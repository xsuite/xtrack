import numpy as np
import xtrack as xt
import xpart as xp

from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import hbar

def _add_polarization_to_tw(tw, line):

    with xt.line._preserve_config(line):

        line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False # For spin

        # Based on:
        # A. Chao, valuation of Radiative Spin Polarization in an Electron Storage Ring
        # https://inspirehep.net/literature/154360

        n0 = np.array([tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]])

        n0 = n0 / np.linalg.norm(n0)

        # Build two orthogonal vectors
        tmp = np.array([0, 0, 1]) # Needs generalization
        l0 = np.cross(n0, tmp)
        l0 = l0 / np.linalg.norm(l0)
        m0 = np.cross(l0, n0)
        m0 = m0 / np.linalg.norm(m0)

        # Build a particle with the three and track them
        p0 = line.build_particles(particle_ref=tw.particle_on_co,
                                mode='shift', x=np.zeros(3))
        p0.spin_x = np.array([l0[0], n0[0], m0[0]])
        p0.spin_y = np.array([l0[1], n0[1], m0[1]])
        p0.spin_z = np.array([l0[2], n0[2], m0[2]])

        line.track(p0, num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
        mon0 = line.record_last_track

        ll = np.zeros((3, len(tw)))
        mm = np.zeros((3, len(tw)))
        nn = np.zeros((3, len(tw)))

        ll[0, :] = mon0.spin_x[0, :]
        ll[1, :] = mon0.spin_y[0, :]
        ll[2, :] = mon0.spin_z[0, :]
        nn[0, :] = mon0.spin_x[1, :]
        nn[1, :] = mon0.spin_y[1, :]
        nn[2, :] = mon0.spin_z[1, :]
        mm[0, :] = mon0.spin_x[2, :]
        mm[1, :] = mon0.spin_y[2, :]
        mm[2, :] = mon0.spin_z[2, :]

        steps_r_matrix = tw.steps_r_matrix

        out = line.compute_one_turn_matrix_finite_differences(particle_on_co=tw.particle_on_co,
                                                            element_by_element=True,
                                                            steps_r_matrix=steps_r_matrix)
        mon_r_ebe = out['mon_ebe']
        part = out['part_temp']

        spin = np.zeros((3, len(part.spin_x)))
        spin[0, :] = part.spin_x
        spin[1, :] = part.spin_y
        spin[2, :] = part.spin_z

        alpha = np.zeros(len(part.spin_x))
        beta = np.zeros(len(part.spin_x))
        for ii in range(len(part.spin_x)):
            alpha[ii] = np.dot(spin[:, ii], l0)
            beta[ii] = np.dot(spin[:, ii], m0)


        steps_r_matrix = out['steps_r_matrix']

        dx = steps_r_matrix["dx"]
        dpx = steps_r_matrix["dpx"]
        dy = steps_r_matrix["dy"]
        dpy = steps_r_matrix["dpy"]
        dzeta = steps_r_matrix["dzeta"]
        ddelta = steps_r_matrix["ddelta"]

        dpzeta = float(part.ptau[6] - part.ptau[12])/2/part.beta0[0]

        temp_mat = np.zeros((2, len(part.spin_x)))
        temp_mat[0, :] = alpha
        temp_mat[1, :] = beta

        DD = np.zeros((2, 6))

        for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, dpzeta]):
            DD[:, jj] = (temp_mat[:, jj+1] - temp_mat[:, jj+1+6])/(2*dd)

        RR = np.eye(8)
        RR[:6, :6] = out['R_matrix']
        RR[6:, :6] = DD

        # Build matrix for alpha/beta discontinuity at end ring
        A_entry = np.zeros((3, 3))
        A_exit = np.zeros((3, 3))

        A_entry[:, 0] = ll[:, 0]
        A_entry[:, 1] = mm[:, 0]
        A_entry[:, 2] = nn[:, 0]
        A_exit[:, 0] = ll[:, -1]
        A_exit[:, 1] = mm[:, -1]
        A_exit[:, 2] = nn[:, -1]

        A_discont = np.linalg.inv(A_entry) @ A_exit
        R_discont = np.eye(8)
        R_discont[6:, 6:] = A_discont[:2, :2]

        R_one_turn = RR @ R_discont

        eival, eivec = np.linalg.eig(R_one_turn)

        ##### Sort modes in pairs of conjugate modes #####
        w0 = eival
        v0 = eivec
        index_list = [0,7,5,1,2,6,3,4] # we mix them up to check the algorithm
        conj_modes = np.zeros([4,2], dtype=np.int64)
        for j in [0,1,2]:
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

        conj_modes[3,0] = index_list[0]
        conj_modes[3,1] = index_list[1]

        modes = np.empty(4, dtype=np.int64)
        modes[0] = conj_modes[0, 0]
        modes[1] = conj_modes[1, 0]
        modes[2] = conj_modes[2, 0]
        modes[3] = conj_modes[3, 0]

        # Sort modes such that (1,2,3,4) is close to (x,y,zeta,spin)
        # Identify the spin mode (the one with no xyz part)
        for i in [0,1,2]:
            if np.linalg.norm(v0[:,modes[3]][:6]) > np.linalg.norm(v0[:,modes[i]][:6]):
                modes[3], modes[i] = modes[i], modes[3]
        # Identify the longitudinal mode
        for i in [0,1]:
            if abs(v0[:,modes[2]])[5] < abs(v0[:,modes[i]])[5]:
                modes[2], modes[i] = modes[i], modes[2]
        # Identify the vertical mode
        if abs(v0[:,modes[1]])[2] < abs(v0[:,modes[0]])[2]:
            modes[0], modes[1] = modes[1], modes[0]

        a1 = v0[:6, modes[0]].real
        a2 = v0[:6, modes[1]].real
        a3 = v0[:6, modes[2]].real
        b1 = v0[:6, modes[0]].imag
        b2 = v0[:6, modes[1]].imag
        b3 = v0[:6, modes[2]].imag

        S = np.array([[0., 1., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0.],
                    [ 0., 0., 0., 1., 0., 0.],
                    [ 0., 0.,-1., 0., 0., 0.],
                    [ 0., 0., 0., 0., 0., 1.],
                    [ 0., 0., 0., 0.,-1., 0.]])

        n1_inv_sq = np.abs(np.matmul(np.matmul(a1, S), b1))
        n2_inv_sq = np.abs(np.matmul(np.matmul(a2, S), b2))
        n3_inv_sq = np.abs(np.matmul(np.matmul(a3, S), b3))

        n1 = 1./np.sqrt(n1_inv_sq)
        n2 = 1./np.sqrt(n2_inv_sq)
        n3 = 1./np.sqrt(n3_inv_sq)

        e1 = eivec[:, modes[0]] * n1
        e2 = eivec[:, modes[1]] * n2
        e3 = eivec[:, modes[2]] * n3

        scale_e1 = np.max([np.abs(e1[0])/dx, np.abs(e1[1])/dpx])
        e1_scaled = e1 / scale_e1
        scale_e2 = np.max([np.abs(e2[2])/dy, np.abs(e2[3])/dpy])
        e2_scaled = e2 / scale_e2
        scale_e3 = np.max([np.abs(e3[4])/dzeta, np.abs(e3[5])/dpzeta])
        e3_scaled = e3 / scale_e3

        e1_trk_re = e1_scaled.real
        e1_trk_im = e1_scaled.imag
        e2_trk_re = e2_scaled.real
        e2_trk_im = e2_scaled.imag
        e3_trk_re = e3_scaled.real
        e3_trk_im = e3_scaled.imag

        e1_spin_re = e1_trk_re[6] * l0 + e1_trk_re[7] * m0 + np.sqrt(1 - e1_trk_re[6]**2 - e1_trk_re[7]**2) * n0
        e1_spin_im = e1_trk_im[6] * l0 + e1_trk_im[7] * m0 + np.sqrt(1 - e1_trk_im[6]**2 - e1_trk_im[7]**2) * n0
        e2_spin_re = e2_trk_re[6] * l0 + e2_trk_re[7] * m0 + np.sqrt(1 - e2_trk_re[6]**2 - e2_trk_re[7]**2) * n0
        e2_spin_im = e2_trk_im[6] * l0 + e2_trk_im[7] * m0 + np.sqrt(1 - e2_trk_im[6]**2 - e2_trk_im[7]**2) * n0
        e3_spin_re = e3_trk_re[6] * l0 + e3_trk_re[7] * m0 + np.sqrt(1 - e3_trk_re[6]**2 - e3_trk_re[7]**2) * n0
        e3_spin_im = e3_trk_im[6] * l0 + e3_trk_im[7] * m0 + np.sqrt(1 - e3_trk_im[6]**2 - e3_trk_im[7]**2) * n0

        x = tw.x[0] + np.array([
            e1_trk_re[0], e1_trk_im[0],
            e2_trk_re[0], e2_trk_im[0],
            e3_trk_re[0], e3_trk_im[0],
        ])
        px = tw.px[0] + np.array([
            e1_trk_re[1], e1_trk_im[1],
            e2_trk_re[1], e2_trk_im[1],
            e3_trk_re[1], e3_trk_im[1],
        ])
        y = tw.y[0] + np.array([
            e1_trk_re[2], e1_trk_im[2],
            e2_trk_re[2], e2_trk_im[2],
            e3_trk_re[2], e3_trk_im[2],
        ])
        py = tw.py[0] + np.array([
            e1_trk_re[3], e1_trk_im[3],
            e2_trk_re[3], e2_trk_im[3],
            e3_trk_re[3], e3_trk_im[3],
        ])
        zeta = tw.zeta[0] + np.array([
            e1_trk_re[4], e1_trk_im[4],
            e2_trk_re[4], e2_trk_im[4],
            e3_trk_re[4], e3_trk_im[4],
        ])
        ptau = tw.ptau[0] + tw.beta0 * np.array([ # in the eigenvector there is pzeta
            e1_trk_re[5], e1_trk_im[5],
            e2_trk_re[5], e2_trk_im[5],
            e3_trk_re[5], e3_trk_im[5],
        ])
        spin_x = np.array([
            e1_spin_re[0], e1_spin_im[0],
            e2_spin_re[0], e2_spin_im[0],
            e3_spin_re[0], e3_spin_im[0],
        ])
        spin_y = np.array([
            e1_spin_re[1], e1_spin_im[1],
            e2_spin_re[1], e2_spin_im[1],
            e3_spin_re[1], e3_spin_im[1],
        ])
        spin_z = np.array([
            e1_spin_re[2], e1_spin_im[2],
            e2_spin_re[2], e2_spin_im[2],
            e3_spin_re[2], e3_spin_im[2],
        ])

        par_track = xp.build_particles(
            particle_ref=tw.particle_on_co, mode='set',
            x=x, px=px, y=y, py=py, zeta=zeta, ptau=ptau,
            spin_x=spin_x, spin_y=spin_y, spin_z=spin_z,
        )

        line.track(par_track, turn_by_turn_monitor='ONE_TURN_EBE')
        mon_ebe = line.record_last_track

        e1_ebe = np.zeros((8, len(tw)), dtype=complex)
        e2_ebe = np.zeros((8, len(tw)), dtype=complex)
        e3_ebe = np.zeros((8, len(tw)), dtype=complex)

        e1_ebe[0, :] = ((mon_ebe.x[0, :] - tw.x[0])
                        + 1j * (mon_ebe.x[1, :] - tw.x[1])) * scale_e1
        e2_ebe[0, :] = ((mon_ebe.x[2, :] - tw.x[0])
                        + 1j * (mon_ebe.x[3, :] - tw.x[1])) * scale_e2
        e3_ebe[0, :] = ((mon_ebe.x[4, :] - tw.x[0])
                        + 1j * (mon_ebe.x[5, :] - tw.x[1])) * scale_e3

        e1_ebe[1, :] = ((mon_ebe.px[0, :] - tw.px[0])
                        + 1j * (mon_ebe.px[1, :] - tw.px[1])) * scale_e1
        e2_ebe[1, :] = ((mon_ebe.px[2, :] - tw.px[0])
                        + 1j * (mon_ebe.px[3, :] - tw.px[1])) * scale_e2
        e3_ebe[1, :] = ((mon_ebe.px[4, :] - tw.px[0])
                        + 1j * (mon_ebe.px[5, :] - tw.px[1])) * scale_e3

        e1_ebe[2, :] = ((mon_ebe.y[0, :] - tw.y[0])
                        + 1j * (mon_ebe.y[1, :] - tw.y[1])) * scale_e1
        e2_ebe[2, :] = ((mon_ebe.y[2, :] - tw.y[0])
                        + 1j * (mon_ebe.y[3, :] - tw.y[1])) * scale_e2
        e3_ebe[2, :] = ((mon_ebe.y[4, :] - tw.y[0])
                        + 1j * (mon_ebe.y[5, :] - tw.y[1])) * scale_e3

        e1_ebe[3, :] = ((mon_ebe.py[0, :] - tw.py[0])
                        + 1j * (mon_ebe.py[1, :] - tw.py[1])) * scale_e1
        e2_ebe[3, :] = ((mon_ebe.py[2, :] - tw.py[0])
                        + 1j * (mon_ebe.py[3, :] - tw.py[1])) * scale_e2
        e3_ebe[3, :] = ((mon_ebe.py[4, :] - tw.py[0])
                        + 1j * (mon_ebe.py[5, :] - tw.py[1])) * scale_e3

        e1_ebe[4, :] = ((mon_ebe.zeta[0, :] - tw.zeta[0])
                        + 1j * (mon_ebe.zeta[1, :] - tw.zeta[1])) * scale_e1
        e2_ebe[4, :] = ((mon_ebe.zeta[2, :] - tw.zeta[0])
                        + 1j * (mon_ebe.zeta[3, :] - tw.zeta[1])) * scale_e2
        e3_ebe[4, :] = ((mon_ebe.zeta[4, :] - tw.zeta[0])
                        + 1j * (mon_ebe.zeta[5, :] - tw.zeta[1])) * scale_e3

        e1_ebe[5, :] = ((mon_ebe.ptau[0, :] - tw.ptau[0])
                        + 1j * (mon_ebe.ptau[1, :] - tw.ptau[1])) / tw.beta0 * scale_e1
        e2_ebe[5, :] = ((mon_ebe.ptau[2, :] - tw.ptau[0])
                        + 1j * (mon_ebe.ptau[3, :] - tw.ptau[1])) / tw.beta0 * scale_e2
        e3_ebe[5, :] = ((mon_ebe.ptau[4, :] - tw.ptau[0])
                        + 1j * (mon_ebe.ptau[5, :] - tw.ptau[1])) / tw.beta0 * scale_e3

        e1_spin = np.zeros((3, len(tw)), dtype=complex)
        e1_spin[0, :] = (mon_ebe.spin_x[0, :] + 1j * mon_ebe.spin_x[1, :]) * scale_e1
        e1_spin[1, :] = (mon_ebe.spin_y[0, :] + 1j * mon_ebe.spin_y[1, :]) * scale_e1
        e1_spin[2, :] = (mon_ebe.spin_z[0, :] + 1j * mon_ebe.spin_z[1, :]) * scale_e1
        e2_spin = np.zeros((3, len(tw)), dtype=complex)
        e2_spin[0, :] = (mon_ebe.spin_x[2, :] + 1j * mon_ebe.spin_x[3, :]) * scale_e2
        e2_spin[1, :] = (mon_ebe.spin_y[2, :] + 1j * mon_ebe.spin_y[3, :]) * scale_e2
        e2_spin[2, :] = (mon_ebe.spin_z[2, :] + 1j * mon_ebe.spin_z[3, :]) * scale_e2
        e3_spin = np.zeros((3, len(tw)), dtype=complex)
        e3_spin[0, :] = (mon_ebe.spin_x[4, :] + 1j * mon_ebe.spin_x[5, :]) * scale_e3
        e3_spin[1, :] = (mon_ebe.spin_y[4, :] + 1j * mon_ebe.spin_y[5, :]) * scale_e3
        e3_spin[2, :] = (mon_ebe.spin_z[4, :] + 1j * mon_ebe.spin_z[5, :]) * scale_e3

        e1_ebe[6, :] = np.sum(e1_spin * ll, axis=0)
        e1_ebe[7, :] = np.sum(e1_spin * mm, axis=0)
        e2_ebe[6, :] = np.sum(e2_spin * ll, axis=0)
        e2_ebe[7, :] = np.sum(e2_spin * mm, axis=0)
        e3_ebe[6, :] = np.sum(e3_spin * ll, axis=0)
        e3_ebe[7, :] = np.sum(e3_spin * mm, axis=0)

        # Rephase
        phix = np.angle(e1_ebe[0, :])
        phiy = np.angle(e2_ebe[2, :])
        phizeta = np.angle(e3_ebe[4, :])

        for ii in range(len(tw)):
            e1_ebe[:, ii] *= np.exp(-1j * phix[ii])
            e2_ebe[:, ii] *= np.exp(-1j * phiy[ii])
            e3_ebe[:, ii] *= np.exp(-1j * phizeta[ii])

        EE = np.zeros((len(tw), 8, 6), complex)
        EE[:, :, 0] = e1_ebe.T
        EE[:, :, 1] = np.conj(e1_ebe.T)
        EE[:, :, 2] = e2_ebe.T
        EE[:, :, 3] = np.conj(e2_ebe.T)
        EE[:, :, 4] = e3_ebe.T
        EE[:, :, 5] = np.conj(e3_ebe.T)

        EE_orb  = EE[:, :6, :]
        EE_spin = EE[:, 6:, :]
        LL = np.real(EE_spin @ np.linalg.inv(EE_orb))

        kin_px = tw.kin_px
        kin_py = tw.kin_py
        delta = tw.delta

        # gamma_dn_dgamma_2 = LL[:, 0, 5] * ll + LL[:, 1, 5] * mm
        gamma_dn_dgamma = -(
            ll * (LL[:, 0, 5]
            # + kin_px / (1 + delta) * LL[:, 0, 1]
            # + kin_py / (1 + delta) * LL[:, 0, 3]
            )
            + mm * (LL[:, 1, 5]
            # + kin_px / (1 + delta) * LL[:, 1, 1]
            # + kin_py / (1 + delta) * LL[:, 1, 3]
            ))

        # Note that here alpha is the l component and beta the m component
        # (opposite on the paper by Chao)
        l_component_e1 = np.imag(np.conj(e1_ebe[4, :]) * e1_ebe[6, :])
        l_component_e2 = np.imag(np.conj(e2_ebe[4, :]) * e2_ebe[6, :])
        l_component_e3 = np.imag(np.conj(e3_ebe[4, :]) * e3_ebe[6, :])
        m_component_e1 = np.imag(np.conj(e1_ebe[4, :]) * e1_ebe[7, :])
        m_component_e2 = np.imag(np.conj(e2_ebe[4, :]) * e2_ebe[7, :])
        m_component_e3 = np.imag(np.conj(e3_ebe[4, :]) * e3_ebe[7, :])

        gamma_dn_dgamma_e1 = np.zeros((3, len(tw)))
        gamma_dn_dgamma_e2 = np.zeros((3, len(tw)))
        gamma_dn_dgamma_e3 = np.zeros((3, len(tw)))

        gamma_dn_dgamma_e1[0, :] = -2 * (l_component_e1 * ll[0, :] + m_component_e1 * mm[0, :])
        gamma_dn_dgamma_e1[1, :] = -2 * (l_component_e1 * ll[1, :] + m_component_e1 * mm[1, :])
        gamma_dn_dgamma_e1[2, :] = -2 * (l_component_e1 * ll[2, :] + m_component_e1 * mm[2, :])
        gamma_dn_dgamma_e2[0, :] = -2 * (l_component_e2 * ll[0, :] + m_component_e2 * mm[0, :])
        gamma_dn_dgamma_e2[1, :] = -2 * (l_component_e2 * ll[1, :] + m_component_e2 * mm[1, :])
        gamma_dn_dgamma_e2[2, :] = -2 * (l_component_e2 * ll[2, :] + m_component_e2 * mm[2, :])
        gamma_dn_dgamma_e3[0, :] = -2 * (l_component_e3 * ll[0, :] + m_component_e3 * mm[0, :])
        gamma_dn_dgamma_e3[1, :] = -2 * (l_component_e3 * ll[1, :] + m_component_e3 * mm[1, :])
        gamma_dn_dgamma_e3[2, :] = -2 * (l_component_e3 * ll[2, :] + m_component_e3 * mm[2, :])

        # PATCH BASED ON BMAD COMPARISON, TO BE UNDERSTOOD!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        gamma_dn_dgamma_e1 *= 0.5
        gamma_dn_dgamma_e2 *= 0.5
        gamma_dn_dgamma_e3 *= 0.5

        gamma_dn_dgamma_2 = gamma_dn_dgamma_e1 + gamma_dn_dgamma_e2 + gamma_dn_dgamma_e3

        gamma_dn_dgamma_mod = np.sqrt(gamma_dn_dgamma[0, :]**2
                                    + gamma_dn_dgamma[1, :]**2
                                    + gamma_dn_dgamma[2, :]**2)

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
        Bz = tw.ks * brho_ref
        B_mod = np.sqrt(Bx**2 + By**2 + Bz**2)
        B_mod[B_mod == 0] = 999. # avoid division by zero

        ib_x = Bx / B_mod
        ib_y = By / B_mod
        ib_z = Bz / B_mod

        n0_ib = tw.spin_x * ib_x + tw.spin_y * ib_y + tw.spin_z * ib_z
        gamma_dn_dgamma_ib = (gamma_dn_dgamma[0, :] * ib_x
                            + gamma_dn_dgamma[1, :] * ib_y
                            + gamma_dn_dgamma[2, :] * ib_z)

        int_kappa3_n0_ib = np.sum(kappa**3 * n0_ib * tw.length)
        int_kappa3_gamma_dn_dgamma_ib = np.sum(kappa**3 * gamma_dn_dgamma_ib * tw.length)
        int_kappa3_11_18_gamma_dn_dgamma_sq = 11./18. * np.sum(kappa**3 * gamma_dn_dgamma_mod**2 * tw.length)

        alpha_minus_co = 1. / tw.circumference * np.sum(kappa**3 * n0_ib *  tw.length)

        alpha_plus_co = 1. / tw.circumference * np.sum(
            kappa**3 * (1 - 2./9. * n0_iv**2) * tw.length)

        alpha_plus = alpha_plus_co + int_kappa3_11_18_gamma_dn_dgamma_sq / tw.circumference
        alpha_minus = alpha_minus_co - int_kappa3_gamma_dn_dgamma_ib / tw.circumference

        pol_inf = 8 / 5 / np.sqrt(3) * alpha_minus_co / alpha_plus_co
        pol_eq = 8 / 5 / np.sqrt(3) * alpha_minus / alpha_plus

        tp_inv = 5 * np.sqrt(3) / 8 * r0 * hbar * tw.gamma0**5 / m0_kg * alpha_plus_co
        tp_s = 1 / tp_inv
        tp_turn = tp_s / tw.T_rev0

        tw._data['alpha_plus_co'] = alpha_plus_co
        tw._data['alpha_minus_co'] = alpha_minus_co
        tw._data['alpha_plus'] = alpha_plus
        tw._data['alpha_minus'] = alpha_minus
        tw['gamma_dn_dgamma_x'] = gamma_dn_dgamma[0, :]
        tw['gamma_dn_dgamma_y'] = gamma_dn_dgamma[1, :]
        tw['gamma_dn_dgamma_z'] = gamma_dn_dgamma[2, :]
        tw['gamma_dn_dgamma_e1'] = gamma_dn_dgamma_e1
        tw['gamma_dn_dgamma_e2'] = gamma_dn_dgamma_e2
        tw['gamma_dn_dgamma_e3'] = gamma_dn_dgamma_e3
        tw['gamma_dn_dgamma_mod'] = gamma_dn_dgamma_mod
        tw['gamma_dn_dgamma'] = gamma_dn_dgamma
        tw['gamma_dn_dgamma_2'] = gamma_dn_dgamma_2
        tw._data['int_kappa3_n0_ib'] = int_kappa3_n0_ib
        tw._data['int_kappa3_gamma_dn_dgamma_ib'] = int_kappa3_gamma_dn_dgamma_ib
        tw._data['int_kappa3_11_18_gamma_dn_dgamma_sq'] = int_kappa3_11_18_gamma_dn_dgamma_sq
        tw._data['pol_inf'] = pol_inf
        tw._data['pol_eq'] = pol_eq
        tw._data['EE'] = EE
        tw['n0_ib'] = n0_ib
        tw['t_pol_turn'] = tp_turn