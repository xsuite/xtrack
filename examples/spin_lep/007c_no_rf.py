import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import m_e
from scipy.constants import hbar

num_turns = 500
bmad = True

line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
line.particle_ref.gamma0 = 89207.78287659843 # to have a spin tune of 103.45
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]

# line.cycle('b2m.qf45.l6', inplace=True)

tt = line.get_table(attr=True)
tt_bend = tt.rows[(tt.element_type == 'RBend') | (tt.element_type == 'Bend')]
tt_quad = tt.rows[tt.element_type == 'Quadrupole']
tt_sext = tt.rows[tt.element_type == 'Sextupole']

# simplify the line to facilitate bmad comparison
for nn in tt_bend.name:
    line[nn].k1 = 0
    line[nn].knl[2] = 0
    line[nn].edge_entry_angle = 0
    line[nn].edge_exit_angle = 0

for nn in tt_sext.name:
    line[nn].k2 = 0

line.set(tt_bend, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)
line.set(tt_quad, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)

line['on_sol.2'] = 1
line['on_sol.4'] = 1
line['on_sol.6'] = 1
line['on_sol.8'] = 1
line['on_spin_bump.2'] = 1
line['on_spin_bump.4'] = 1
line['on_spin_bump.6'] = 1
line['on_spin_bump.8'] = 1
line['on_coupl_sol.2'] = 0
line['on_coupl_sol.4'] = 0
line['on_coupl_sol.6'] = 0
line['on_coupl_sol.8'] = 0
line['on_coupl_sol_bump.2'] = 0
line['on_coupl_sol_bump.4'] = 0
line['on_coupl_sol_bump.6'] = 0
line['on_coupl_sol_bump.8'] = 0

if bmad:
    from bmad_track_twiss_spin import bmad_run
    bmad_data = bmad_run(line)
    df = bmad_data['spin']
    df_orb = bmad_data['optics']
    spin_summary_bmad = bmad_data['spin_summary']

tw = line.twiss4d(spin=True, radiation_integrals=True)

line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False # For spin

# Based on:
# A. Chao, valuation of Radiative Spin Polarization in an Electron Storage Ring
# https://inspirehep.net/literature/154360

steps_r_matrix = tw.steps_r_matrix

for kk in steps_r_matrix:
    steps_r_matrix[kk] *= 0.1

out = line.compute_one_turn_matrix_finite_differences(particle_on_co=tw.particle_on_co,
                                                    element_by_element=True,
                                                    steps_r_matrix=steps_r_matrix)
mon_r_ebe = out['mon_ebe']
part = out['part_temp']

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
RR_orb = out['R_matrix'].copy()
RR[:6, :6] = out['R_matrix']
RR[6:, :6] = DD

# Spin response matrix
ds = 1e-5

p_test = xp.build_particles(particle_ref=tw.particle_on_co, mode='shift',
                            x=[0,0,0,0,0,0])
p_test.spin_x = [ds, 0, 0, -ds, 0, 0]
p_test.spin_y = [0, ds, 0, 0, -ds, 0]
p_test.spin_z = [0, 0, ds, 0, 0, -ds]

line.track(p_test)

A = np.zeros((3, 3))
A[0, 0] = (p_test.spin_x[0] - p_test.spin_x[3])/(2*ds)
A[0, 1] = (p_test.spin_x[1] - p_test.spin_x[4])/(2*ds)
A[0, 2] = (p_test.spin_x[2] - p_test.spin_x[5])/(2*ds)
A[1, 0] = (p_test.spin_y[0] - p_test.spin_y[3])/(2*ds)
A[1, 1] = (p_test.spin_y[1] - p_test.spin_y[4])/(2*ds)
A[1, 2] = (p_test.spin_y[2] - p_test.spin_y[5])/(2*ds)
A[2, 0] = (p_test.spin_z[0] - p_test.spin_z[3])/(2*ds)
A[2, 1] = (p_test.spin_z[1] - p_test.spin_z[4])/(2*ds)
A[2, 2] = (p_test.spin_z[2] - p_test.spin_z[5])/(2*ds)

RR[6:, 6:] = A

RR_reduced = np.delete(np.delete(RR, 4, axis=0), 4, axis=1)
eival_all, eivec_all = np.linalg.eig(RR_reduced)

# Suppress the 4th row and col
RR_orb = np.delete(RR_orb, 4, axis=0)
RR_orb = np.delete(RR_orb, 4, axis=1)

eival, EE_orb = np.linalg.eig(RR_orb)
n_eigen = EE_orb.shape[1]

# Add a dummy row 4 in eivec
EE_orb = np.insert(EE_orb, 4, 0, axis=0)

EE_spin = np.zeros((3, n_eigen), dtype=complex)
for ii in range(n_eigen):
    # EE_spin[:, ii] = DD @ EE_orb[:, ii]
    EE_spin[:, ii] = np.linalg.inv(eival[ii] * np.eye(3) - A) @ DD @ EE_orb[:, ii]


eee = np.zeros((9, n_eigen), dtype=complex)
eee[:6, :] = EE_orb
eee[6:, :] = EE_spin

# Identify eigenvector with eigenvalue 1 and remove n0 component
i_eigen_one = np.argmin(np.abs(eival - 1))
n0 = np.array([tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]])
eee[6:, i_eigen_one] -= np.dot(eee[6:, i_eigen_one], n0) * n0

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
    for ii, key in enumerate(['x', 'px', 'y', 'py', 'zeta', 'ptau',
                              'spin_x', 'spin_y', 'spin_z']):
        particle_data[key] = tw[key][0] + np.array(
            list(eee_trk_re[ii, :]) + list(eee_trk_im[ii, :])
        )

    par_track = xp.build_particles(
        particle_ref=tw.particle_on_co, mode='set', **particle_data
    )

    line.track(par_track, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_ebe = line.record_last_track

    ee_ebe = np.zeros((len(tw), 9, n_eigen), dtype=complex)

    for ii, key in enumerate(['x', 'px', 'y', 'py', 'zeta', 'ptau',
                              'spin_x', 'spin_y', 'spin_z']):
        mon_vv = getattr(mon_ebe, key)
        for iee in range(n_eigen):
            ee_ebe[:, ii, iee] = side *((mon_vv[iee, :] - tw[key])
                            + 1j * (mon_vv[n_eigen + iee, :] - tw[key]))

    # Rephase
    for ii in range(n_eigen):
        i_max = np.argmax(np.abs(ee_ebe[0, :, ii])) # Strongest component at start ring
        this_phi = np.angle(ee_ebe[:, i_max, ii])
        for jj in range(ee_ebe.shape[1]):
            ee_ebe[:, jj, ii] *= np.exp(-1j * this_phi)

    EE = ee_ebe.copy()

    EE_side[side] = EE

# Average the two sides
EE = 0.5 * (EE_side[1] + EE_side[-1])
EE_orb  = EE[:, :6, :]
EE_spin = EE[:, 6:, :]

# # Transform to x', y'
# kin_px_co = tw.kin_px
# kin_py_co = tw.kin_py
# delta = tw.delta
# TT = np.zeros((len(tw), 6, 6))
# TT[:, 0, 0] = 1
# TT[:, 1, 1] = (1 - delta)
# TT[:, 1, 5] = -kin_px_co
# TT[:, 2, 2] = 1
# TT[:, 3, 3] = (1 - delta)
# TT[:, 3, 5] = -kin_py_co
# TT[:, 4, 4] = 1
# TT[:, 5, 5] = 1
# EE_orb = TT @ EE_orb

# Remove the 4th row
EE_orb = np.delete(EE_orb, 4, axis=1)
filter = np.diag([1, 1, 1, 1, 1])
LL = np.real(EE_spin @ filter @ np.linalg.inv(EE_orb))
# Add a dummy col 4 in LL
LL = np.insert(LL, 4, 0, axis=2)
gamma_dn_dgamma = LL[:, :, 5]

# Should be equivalent...
# # Remove the 4th row
# EE_orb = np.delete(EE_orb, 4, axis=1)
# ene_one = np.zeros((len(tw), 5), dtype=complex)
# ene_one[:, 4] = 1.
# a_coeff = 0 * ene_one
# gamma_dn_dgamma = np.zeros((len(tw), 3), dtype=complex)
# for ii in range(len(tw)):
#     a_coeff[ii, :] = np.linalg.solve(EE_orb[ii, :, :], ene_one[ii, :])
#     gamma_dn_dgamma[ii, :] = EE_spin[ii, :, :] @ a_coeff[ii, :]

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

dny_ref = 1 / (np.sqrt(1 - tw.spin_x**2 - tw.spin_z**2)) * (
    -tw.spin_x * tw.gamma_dn_dgamma[:, 0] - tw.spin_z * tw.gamma_dn_dgamma[:, 2])

print('Xsuite polarization: ', tw.pol_eq)

if bmad:
    print('Bmad polarization:   ', spin_summary_bmad['Polarization Limit DK'])

import matplotlib.pyplot as plt
plt.close('all')

if bmad and df.spin_y[0] < 0:
    for kk in ['spin_x', 'spin_y', 'spin_z',
                'spin_dn_dpz_x', 'spin_dn_dpz_y', 'spin_dn_dpz_z']:
          df[kk] *= -1

dn_dy_ref_bmad = 1 / np.sqrt(1 - df.spin_x**2 - df.spin_z**2) * (
    -df.spin_x * df.spin_dn_dpz_x - df.spin_z * df.spin_dn_dpz_z)

i_eig = 0
two_plus = line.twiss(spin=True,
                 betx=0, bety=0,
                 x=eee_trk_re[0, i_eig] + tw.x[0],
                 px=eee_trk_re[1, i_eig] + tw.px[0],
                 y=eee_trk_re[2, i_eig] + tw.y[0],
                 py=eee_trk_re[3, i_eig] + tw.py[0],
                 zeta=eee_trk_re[4, i_eig] + tw.zeta[0],
                 delta=eee_trk_re[5, i_eig] + tw.delta[0],
                 spin_x=eee_trk_re[6, i_eig] + tw.spin_x[0],
                 spin_y=eee_trk_re[7, i_eig] + tw.spin_y[0],
                 spin_z=eee_trk_re[8, i_eig] + tw.spin_z[0])
outbmad_plus = bmad_run(line, track=dict(
    x=two_plus.x[0],
    px=two_plus.px[0],
    y=two_plus.y[0],
    py=two_plus.py[0],
    zeta=two_plus.zeta[0],
    delta=two_plus.delta[0],
    spin_x=two_plus.spin_x[0],
    spin_y=two_plus.spin_y[0],
    spin_z=two_plus.spin_z[0]))

two_minus = line.twiss(spin=True,
                 betx=0, bety=0,
                 x=-eee_trk_re[0, i_eig] + tw.x[0],
                 px=-eee_trk_re[1, i_eig] + tw.px[0],
                 y=-eee_trk_re[2, i_eig] + tw.y[0],
                 py=-eee_trk_re[3, i_eig] + tw.py[0],
                 zeta=-eee_trk_re[4, i_eig] + tw.zeta[0],
                 delta=-eee_trk_re[5, i_eig] + tw.delta[0],
                 spin_x=-eee_trk_re[6, i_eig] + tw.spin_x[0],
                 spin_y=-eee_trk_re[7, i_eig] + tw.spin_y[0],
                 spin_z=-eee_trk_re[8, i_eig] + tw.spin_z[0])
outbmad_minus = bmad_run(line, track=dict(
    x=two_minus.x[0],
    px=two_minus.px[0],
    y=two_minus.y[0],
    py=two_minus.py[0],
    zeta=two_minus.zeta[0],
    delta=two_minus.delta[0],
    spin_x=two_minus.spin_x[0],
    spin_y=two_minus.spin_y[0],
    spin_z=two_minus.spin_z[0]))

plt.figure(100)
ax = plt.subplot(2, 1, 1)
plt.plot(df.s, outbmad_plus['optics']['x'] - outbmad_minus['optics']['x'])
plt.plot(two_plus.s, two_plus.x - two_minus.x)
plt.ylabel('x')
ax = plt.subplot(2, 1, 2, sharex=ax)
plt.plot(df.s, outbmad_plus['spin']['spin_x'] - outbmad_minus['spin']['spin_x'])
plt.plot(two_plus.s, two_plus.spin_x - two_minus.spin_x)
plt.ylabel('spin_x')


plt.figure(1)
ax1 = plt.subplot(3, 1, 1)
plt.plot(tw.s, tw.spin_x, label='Xsuite')
if bmad: plt.plot(df.s, df.spin_x, label='Bmad')
plt.ylabel('spin_x')
plt.legend()
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(tw.s, tw.spin_y, label='y')
if bmad: plt.plot(df.s, df.spin_y, label='y Bmad')
plt.ylabel('spin_y')
plt.legend()
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.xlabel('s [m]')
plt.plot(tw.s, tw.spin_z, label='z')
if bmad: plt.plot(df.s, df.spin_z, label='z Bmad')
plt.ylabel('spin_z')
plt.legend()

plt.figure(2, figsize=(12, 6))
plt.subplot(3, 2, 1)
plt.plot(tw.s, tw.EE_side[1][:, 7, 0].real, label='+ re')
plt.plot(tw.s, tw.EE_side[-1][:, 7, 0].real, label='- re')
plt.plot(tw.s, tw.EE_side[1][:, 7, 0].imag, label='+ im')
plt.plot(tw.s, tw.EE_side[-1][:, 7, 0].imag, label='- im')
plt.ylabel('e1_ebe')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(tw.s, tw.EE_side[1][:, 7, 1].real, label='+ re')
plt.plot(tw.s, tw.EE_side[-1][:, 7, 1].real, label='- re')
plt.plot(tw.s, tw.EE_side[1][:, 7, 1].imag, label='+ im')
plt.plot(tw.s, tw.EE_side[-1][:, 7, 1].imag, label='- im')
plt.ylabel('e2_ebe')
plt.legend()
plt.subplot(3, 2, 3)
plt.plot(tw.s, tw.EE_side[1][:, 7, 2].real, label='+ re')
plt.plot(tw.s, tw.EE_side[-1][:, 7, 2].real, label='- re')
plt.plot(tw.s, tw.EE_side[1][:, 7, 2].imag, label='+ im')
plt.plot(tw.s, tw.EE_side[-1][:, 7, 2].imag, label='- im')
plt.ylabel('e3_ebe')
plt.subplot(3, 2, 4)
plt.plot(tw.s, tw.EE_side[1][:, 7, 3].real, label='+ re')
plt.plot(tw.s, tw.EE_side[-1][:, 7, 3].real, label='- re')
plt.plot(tw.s, tw.EE_side[1][:, 7, 3].imag, label='+ im')
plt.plot(tw.s, tw.EE_side[-1][:, 7, 3].imag, label='- im')
plt.ylabel('e4_ebe')
plt.subplot(3, 2, 5)
plt.plot(tw.s, tw.EE_side[1][:, 7, 4].real, label='+ re')
plt.plot(tw.s, tw.EE_side[-1][:, 7, 4].real, label='- re')
plt.plot(tw.s, tw.EE_side[1][:, 7, 4].imag, label='+ im')
plt.plot(tw.s, tw.EE_side[-1][:, 7, 4].imag, label='- im')
plt.ylabel('e5_ebe')
if n_eigen > 5:
    plt.subplot(3, 2, 6)
    plt.plot(tw.s, tw.EE_side[1][:, 7, 5].real, label='+ re')
    plt.plot(tw.s, tw.EE_side[-1][:, 7, 5].real, label='- re')
    plt.plot(tw.s, tw.EE_side[1][:, 7, 5].imag, label='+ im')
    plt.plot(tw.s, tw.EE_side[-1][:, 7, 5].imag, label='- im')
    plt.ylabel('e6_ebe')
    plt.xlabel('s [m]')


plt.figure(3, figsize=(8, 6))
ax1 = plt.subplot(3, 1, 1)
tw.plot(lattice_only=True, ax=ax1)
plt.plot(tw.s, tw.gamma_dn_dgamma[:, 0])
plt.plot(df.s, df.spin_dn_dpz_x)
plt.ylabel('gamma_dn_dgamma_x')
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
tw.plot(lattice_only=True, ax=ax2)
plt.plot(tw.s, tw.gamma_dn_dgamma[:, 1])
plt.plot(df.s, df.spin_dn_dpz_y)
plt.plot(tw.s, dny_ref, label='dn_y_ref')
plt.plot(df.s, dn_dy_ref_bmad, label='dn_y_ref_bmad')
plt.ylabel('gamma_dn_dgamma_y')
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
tw.plot(lattice_only=True, ax=ax3)
plt.plot(tw.s, tw.gamma_dn_dgamma[:, 2])
plt.plot(df.s, df.spin_dn_dpz_z)
plt.ylabel('gamma_dn_dgamma_z')
plt.xlabel('s [m]')


plt.show()
