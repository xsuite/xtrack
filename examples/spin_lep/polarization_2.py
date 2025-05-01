import numpy as np
import xtrack as xt
import xpart as xp

from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import hbar

S = np.array([[0., 1., 0., 0., 0., 0.],
            [-1., 0., 0., 0., 0., 0.],
            [ 0., 0., 0., 1., 0., 0.],
            [ 0., 0.,-1., 0., 0., 0.],
            [ 0., 0., 0., 0., 0., 1.],
            [ 0., 0., 0., 0.,-1., 0.]])

def _add_polarization_to_tw(tw, line):

    with xt.line._preserve_config(line):

        line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False # For spin

        # Based on:
        # A. Chao, valuation of Radiative Spin Polarization in an Electron Storage Ring
        # https://inspirehep.net/literature/154360

        steps_r_matrix = tw.steps_r_matrix

        # for kk in steps_r_matrix:
        #     steps_r_matrix[kk] *= 0.1

        out = line.compute_one_turn_matrix_finite_differences(particle_on_co=tw.particle_on_co,
                                                            element_by_element=True,
                                                            steps_r_matrix=steps_r_matrix)
        mon_r_ebe = out['mon_ebe']
        part = out['part_temp']

        spin = np.zeros((3, len(part.spin_x)))
        spin[0, :] = part.spin_x
        spin[1, :] = part.spin_y
        spin[2, :] = part.spin_z

        steps_r_matrix = out['steps_r_matrix']

        dx = steps_r_matrix["dx"]
        dpx = steps_r_matrix["dpx"]
        dy = steps_r_matrix["dy"]
        dpy = steps_r_matrix["dpy"]
        dzeta = steps_r_matrix["dzeta"]
        ddelta = steps_r_matrix["ddelta"]

        dpzeta = float(part.ptau[6] - part.ptau[12])/2/part.beta0[0]

        temp_mat = np.zeros((3, len(part.spin_x)))
        temp_mat[0, :] = part.spin_x
        temp_mat[1, :] = part.spin_y
        temp_mat[2, :] = part.spin_z

        DD = np.zeros((3, 6))

        for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, dpzeta]):
            DD[:, jj] = (temp_mat[:, jj+1] - temp_mat[:, jj+1+6])/(2*dd)

        RR = np.eye(9)
        RR[:6, :6] = out['R_matrix']
        RR[6:, :6] = DD

        R_one_turn = RR

        eival, eivec = np.linalg.eig(R_one_turn)

        # Identify spin modes and remove them
        norm_orbital_part = []
        for ii in range(9):
            norm_orbital_part.append(np.linalg.norm(eivec[:6, ii]))
        i_sorted = np.argsort(norm_orbital_part)
        v0 = eivec[:, i_sorted[3:]]
        w0 = eival[i_sorted[3:]]

        breakpoint()

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
            if np.matmul(np.matmul(a0[:6,ind[0]], S), b0[:6,ind[0]]) > 0:
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

        a1 = v0[:6, modes[0]].real
        a2 = v0[:6, modes[1]].real
        a3 = v0[:6, modes[2]].real
        b1 = v0[:6, modes[0]].imag
        b2 = v0[:6, modes[1]].imag
        b3 = v0[:6, modes[2]].imag

        n1_inv_sq = np.abs(np.matmul(np.matmul(a1, S), b1))
        n2_inv_sq = np.abs(np.matmul(np.matmul(a2, S), b2))
        n3_inv_sq = np.abs(np.matmul(np.matmul(a3, S), b3))

        n1 = 1./np.sqrt(n1_inv_sq)
        n2 = 1./np.sqrt(n2_inv_sq)
        n3 = 1./np.sqrt(n3_inv_sq)

        e1 = v0[:, modes[0]] * n1
        e2 = v0[:, modes[1]] * n2
        e3 = v0[:, modes[2]] * n3

        scale_e1 = np.max([np.abs(e1[0])/dx, np.abs(e1[1])/dpx])
        e1_scaled = e1 / scale_e1
        scale_e2 = np.max([np.abs(e2[2])/dy, np.abs(e2[3])/dpy])
        e2_scaled = e2 / scale_e2
        scale_e3 = np.max([np.abs(e3[4])/dzeta, np.abs(e3[5])/dpzeta])
        e3_scaled = e3 / scale_e3

        breakpoint()

        EE_side = {}

        for side in [1, -1]:

            e1_ebe = np.zeros((9, len(tw)), dtype=complex)
            e2_ebe = np.zeros((9, len(tw)), dtype=complex)
            e3_ebe = np.zeros((9, len(tw)), dtype=complex)

            e1_trk_re = side * e1_scaled.real
            e1_trk_im = side * e1_scaled.imag
            e2_trk_re = side * e2_scaled.real
            e2_trk_im = side * e2_scaled.imag
            e3_trk_re = side * e3_scaled.real
            e3_trk_im = side * e3_scaled.imag

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
            spin_x = tw.spin_x[0] + np.array([
                e1_trk_re[6], e1_trk_im[6],
                e2_trk_re[6], e2_trk_im[6],
                e3_trk_re[6], e3_trk_im[6],
            ])

            spin_y = tw.spin_y[0] + np.array([
                e1_trk_re[7], e1_trk_im[7],
                e2_trk_re[7], e2_trk_im[7],
                e3_trk_re[7], e3_trk_im[7],
            ])

            spin_z = tw.spin_z[0] + np.array([
                e1_trk_re[8], e1_trk_im[8],
                e2_trk_re[8], e2_trk_im[8],
                e3_trk_re[8], e3_trk_im[8],
            ])

            par_track = xp.build_particles(
                particle_ref=tw.particle_on_co, mode='set',
                x=x, px=px, y=y, py=py, zeta=zeta, ptau=ptau,
                spin_x=spin_x, spin_y=spin_y, spin_z=spin_z,
            )

            line.track(par_track, turn_by_turn_monitor='ONE_TURN_EBE')
            mon_ebe = line.record_last_track

            e1_ebe[0, :] = side *((mon_ebe.x[0, :] - tw.x)
                            + 1j * (mon_ebe.x[1, :] - tw.x)) * scale_e1
            e2_ebe[0, :] = side *((mon_ebe.x[2, :] - tw.x)
                            + 1j * (mon_ebe.x[3, :] - tw.x)) * scale_e2
            e3_ebe[0, :] = side *((mon_ebe.x[4, :] - tw.x)
                            + 1j * (mon_ebe.x[5, :] - tw.x)) * scale_e3

            e1_ebe[1, :] = side *((mon_ebe.px[0, :] - tw.px)
                            + 1j * (mon_ebe.px[1, :] - tw.px)) * scale_e1
            e2_ebe[1, :] = side *((mon_ebe.px[2, :] - tw.px)
                            + 1j * (mon_ebe.px[3, :] - tw.px)) * scale_e2
            e3_ebe[1, :] = side *((mon_ebe.px[4, :] - tw.px)
                            + 1j * (mon_ebe.px[5, :] - tw.px)) * scale_e3

            e1_ebe[2, :] = side *((mon_ebe.y[0, :] - tw.y)
                            + 1j * (mon_ebe.y[1, :] - tw.y)) * scale_e1
            e2_ebe[2, :] = side *((mon_ebe.y[2, :] - tw.y)
                            + 1j * (mon_ebe.y[3, :] - tw.y)) * scale_e2
            e3_ebe[2, :] = side *((mon_ebe.y[4, :] - tw.y)
                            + 1j * (mon_ebe.y[5, :] - tw.y)) * scale_e3

            e1_ebe[3, :] = side *((mon_ebe.py[0, :] - tw.py)
                            + 1j * (mon_ebe.py[1, :] - tw.py)) * scale_e1
            e2_ebe[3, :] = side *((mon_ebe.py[2, :] - tw.py)
                            + 1j * (mon_ebe.py[3, :] - tw.py)) * scale_e2
            e3_ebe[3, :] = side *((mon_ebe.py[4, :] - tw.py)
                            + 1j * (mon_ebe.py[5, :] - tw.py)) * scale_e3

            e1_ebe[4, :] = side *((mon_ebe.zeta[0, :] - tw.zeta)
                            + 1j * (mon_ebe.zeta[1, :] - tw.zeta)) * scale_e1
            e2_ebe[4, :] = side *((mon_ebe.zeta[2, :] - tw.zeta)
                            + 1j * (mon_ebe.zeta[3, :] - tw.zeta)) * scale_e2
            e3_ebe[4, :] = side *((mon_ebe.zeta[4, :] - tw.zeta)
                            + 1j * (mon_ebe.zeta[5, :] - tw.zeta)) * scale_e3

            e1_ebe[5, :] = side *((mon_ebe.ptau[0, :] - tw.ptau)
                            + 1j * (mon_ebe.ptau[1, :] - tw.ptau)) / tw.beta0 * scale_e1
            e2_ebe[5, :] = side *((mon_ebe.ptau[2, :] - tw.ptau)
                            + 1j * (mon_ebe.ptau[3, :] - tw.ptau)) / tw.beta0 * scale_e2
            e3_ebe[5, :] = side *((mon_ebe.ptau[4, :] - tw.ptau)
                            + 1j * (mon_ebe.ptau[5, :] - tw.ptau)) / tw.beta0 * scale_e3

            e1_ebe[6, :] = side *((mon_ebe.spin_x[0, :] - tw.spin_x)
                            + 1j * (mon_ebe.spin_x[1, :] - tw.spin_x)) * scale_e1
            e2_ebe[6, :] = side *((mon_ebe.spin_x[2, :] - tw.spin_x)
                            + 1j * (mon_ebe.spin_x[3, :] - tw.spin_x)) * scale_e2
            e3_ebe[6, :] = side *((mon_ebe.spin_x[4, :] - tw.spin_x)
                            + 1j * (mon_ebe.spin_x[5, :] - tw.spin_x)) * scale_e3

            e1_ebe[7, :] = side *((mon_ebe.spin_y[0, :] - tw.spin_y)
                            + 1j * (mon_ebe.spin_y[1, :] - tw.spin_y)) * scale_e1
            e2_ebe[7, :] = side *((mon_ebe.spin_y[2, :] - tw.spin_y)
                            + 1j * (mon_ebe.spin_y[3, :] - tw.spin_y)) * scale_e2
            e3_ebe[7, :] = side *((mon_ebe.spin_y[4, :] - tw.spin_y)
                            + 1j * (mon_ebe.spin_y[5, :] - tw.spin_y)) * scale_e3

            e1_ebe[8, :] = side *((mon_ebe.spin_z[0, :] - tw.spin_z)
                            + 1j * (mon_ebe.spin_z[1, :] - tw.spin_z)) * scale_e1
            e2_ebe[8, :] = side *((mon_ebe.spin_z[2, :] - tw.spin_z)
                            + 1j * (mon_ebe.spin_z[3, :] - tw.spin_z)) * scale_e2
            e3_ebe[8, :] = side *((mon_ebe.spin_z[4, :] - tw.spin_z)
                            + 1j * (mon_ebe.spin_z[5, :] - tw.spin_z)) * scale_e3

            # Rephase
            phix = np.angle(e1_ebe[0, :])
            phiy = np.angle(e2_ebe[2, :])
            phizeta = np.angle(e3_ebe[4, :])

            for ii in range(len(tw)):
                e1_ebe[:, ii] *= np.exp(-1j * phix[ii])
                e2_ebe[:, ii] *= np.exp(-1j * phiy[ii])
                e3_ebe[:, ii] *= np.exp(-1j * phizeta[ii])

            EE = np.zeros((len(tw), 9, 6), complex)
            EE[:, :, 0] = e1_ebe.T
            EE[:, :, 1] = np.conj(e1_ebe.T)
            EE[:, :, 2] = e2_ebe.T
            EE[:, :, 3] = np.conj(e2_ebe.T)
            EE[:, :, 4] = e3_ebe.T
            EE[:, :, 5] = np.conj(e3_ebe.T)

            EE_side[side] = EE

        EE = 0.5 * (EE_side[1] + EE_side[-1])
        EE_orb  = EE[:, :6, :]
        EE_spin = EE[:, 6:, :]
        LL = np.zeros([len(tw), 3, 6], dtype=float)

        # np.real(EE_spin @ np.linalg.inv(EE_orb))

        kin_px = tw.kin_px
        kin_py = tw.kin_py
        delta = tw.delta

        gamma_dn_dgamma = LL[:, :, 5]

        gamma_dn_dgamma_mod = np.sqrt(gamma_dn_dgamma[:, 0]**2
                                    + gamma_dn_dgamma[:, 1]**2
                                    + gamma_dn_dgamma[:, 2]**2)

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
        gamma_dn_dgamma_ib = (gamma_dn_dgamma[:, 0] * ib_x
                            + gamma_dn_dgamma[:, 1] * ib_y
                            + gamma_dn_dgamma[:, 2] * ib_z)

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
        tw['gamma_dn_dgamma_mod'] = gamma_dn_dgamma_mod
        tw['gamma_dn_dgamma'] = gamma_dn_dgamma
        tw._data['int_kappa3_n0_ib'] = int_kappa3_n0_ib
        tw._data['int_kappa3_gamma_dn_dgamma_ib'] = int_kappa3_gamma_dn_dgamma_ib
        tw._data['int_kappa3_11_18_gamma_dn_dgamma_sq'] = int_kappa3_11_18_gamma_dn_dgamma_sq
        tw._data['pol_inf'] = pol_inf
        tw._data['pol_eq'] = pol_eq
        tw._data['EE'] = EE
        tw._data['EE_side'] = EE_side
        tw['n0_ib'] = n0_ib
        tw['t_pol_turn'] = tp_turn
