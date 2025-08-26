"""
To intall bmad:
  conda install -c conda-forge bmad
  pip install pytao
"""

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

vrfc231 = 0.
method = '4d'

# vrfc231 = 12.65 # qs=0.6
# method = '6d'

line = xt.load('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
line.particle_ref.gamma0 = 89207.78287659843 # to have a spin tune of 103.45
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]
line['vrfc231'] = vrfc231

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

# for nn in tt_sext.name:
#     line[nn].k2 = 0

line.set(tt_bend, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)
line.set(tt_quad, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)

line['on_solenoids'] = 1
line['on_spin_bumps'] = 1
line['on_coupling_corrections'] = 1

if bmad:
    from bmad_track_twiss_spin import bmad_run
    bmad_data = bmad_run(line)
    df = bmad_data['spin']
    df_orb = bmad_data['optics']
    spin_summary_bmad = bmad_data['spin_summary']

tw = line.twiss4d(polarization=True)

print('Xsuite polarization: ', tw.spin_polarization_eq)

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
eee_trk_re = tw._spin_eee_trk_re
eee_trk_im = tw._spin_eee_trk_im
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

plt.figure(100, figsize=(6.4, 4.8*1.5))
ax = plt.subplot(5, 1, 1)
tw.plot(lattice_only=True, ax=ax)
plt.plot(df.s, outbmad_plus['optics']['x'] - outbmad_minus['optics']['x'])
plt.plot(two_plus.s, two_plus.x - two_minus.x)
plt.ylabel('x')
ax = plt.subplot(5, 1, 2, sharex=ax)
plt.plot(df.s, outbmad_plus['optics']['y'] - outbmad_minus['optics']['y'])
plt.plot(two_plus.s, two_plus.y - two_minus.y)
plt.ylabel('y')
ax = plt.subplot(5, 1, 3, sharex=ax)
plt.plot(df.s, outbmad_plus['spin']['spin_x'] - outbmad_minus['spin']['spin_x'])
plt.plot(two_plus.s, two_plus.spin_x - two_minus.spin_x)
plt.ylabel('spin_x')
ax = plt.subplot(5, 1, 4, sharex=ax)
plt.plot(df.s, outbmad_plus['spin']['spin_y'] - outbmad_minus['spin']['spin_y'])
plt.plot(two_plus.s, two_plus.spin_y - two_minus.spin_y)
plt.ylabel('spin_y')
ax = plt.subplot(5, 1, 5, sharex=ax)
plt.plot(df.s, outbmad_plus['spin']['spin_z'] - outbmad_minus['spin']['spin_z'])
plt.plot(two_plus.s, two_plus.spin_z - two_minus.spin_z)
plt.ylabel('spin_z')
plt.xlabel('s [m]')

plt.figure(1, figsize=(6.4*1.1, 4.8*1.6))
ax0 = plt.subplot(4, 1, 1)
plt.plot(tw.s, tw.y, label='y')
plt.ylabel('Closed orbit y [m]')
ax1 = plt.subplot(4, 1, 2, sharex=ax0)
plt.plot(tw.s, tw.spin_x, label='Xsuite')
if bmad: plt.plot(df.s, df.spin_x, '--', label='Bmad')
plt.ylabel('spin_x')
# plt.legend()
ax2 = plt.subplot(4, 1, 3, sharex=ax1)
plt.plot(tw.s, tw.spin_y, label='Xsuite')
if bmad: plt.plot(df.s, df.spin_y, '--', label='Bmad')
plt.ylabel('spin_y')
plt.legend()
ax3 = plt.subplot(4, 1, 4, sharex=ax1)
plt.xlabel('s [m]')
plt.plot(tw.s, tw.spin_z, label='z')
if bmad: plt.plot(df.s, df.spin_z, '--',  label='z Bmad')
plt.ylabel('spin_z')
plt.subplots_adjust(hspace=.26, left=.15, bottom=0.08, top=0.95)
# plt.legend()

EE_side = tw._spin_ee_side
plt.figure(2, figsize=(12, 6))
plt.subplot(3, 2, 1)
plt.plot(tw.s, EE_side[1][:, 7, 0].real, label='+ re')
plt.plot(tw.s, EE_side[-1][:, 7, 0].real, label='- re')
plt.plot(tw.s, EE_side[1][:, 7, 0].imag, label='+ im')
plt.plot(tw.s, EE_side[-1][:, 7, 0].imag, label='- im')
plt.ylabel('e1_ebe')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(tw.s, EE_side[1][:, 7, 1].real, label='+ re')
plt.plot(tw.s, EE_side[-1][:, 7, 1].real, label='- re')
plt.plot(tw.s, EE_side[1][:, 7, 1].imag, label='+ im')
plt.plot(tw.s, EE_side[-1][:, 7, 1].imag, label='- im')
plt.ylabel('e2_ebe')
plt.legend()
plt.subplot(3, 2, 3)
plt.plot(tw.s, EE_side[1][:, 7, 2].real, label='+ re')
plt.plot(tw.s, EE_side[-1][:, 7, 2].real, label='- re')
plt.plot(tw.s, EE_side[1][:, 7, 2].imag, label='+ im')
plt.plot(tw.s, EE_side[-1][:, 7, 2].imag, label='- im')
plt.ylabel('e3_ebe')
plt.subplot(3, 2, 4)
plt.plot(tw.s, EE_side[1][:, 7, 3].real, label='+ re')
plt.plot(tw.s, EE_side[-1][:, 7, 3].real, label='- re')
plt.plot(tw.s, EE_side[1][:, 7, 3].imag, label='+ im')
plt.plot(tw.s, EE_side[-1][:, 7, 3].imag, label='- im')
plt.ylabel('e4_ebe')
plt.subplot(3, 2, 5)
plt.plot(tw.s, EE_side[1][:, 7, 4].real, label='+ re')
plt.plot(tw.s, EE_side[-1][:, 7, 4].real, label='- re')
plt.plot(tw.s, EE_side[1][:, 7, 4].imag, label='+ im')
plt.plot(tw.s, EE_side[-1][:, 7, 4].imag, label='- im')
plt.ylabel('e5_ebe')
n_eigen = tw.spin_eigenvectors.shape[-1]
if n_eigen > 5:
    plt.subplot(3, 2, 6)
    plt.plot(tw.s, EE_side[1][:, 7, 5].real, label='+ re')
    plt.plot(tw.s, EE_side[-1][:, 7, 5].real, label='- re')
    plt.plot(tw.s, EE_side[1][:, 7, 5].imag, label='+ im')
    plt.plot(tw.s, EE_side[-1][:, 7, 5].imag, label='- im')
    plt.ylabel('e6_ebe')
    plt.xlabel('s [m]')


plt.figure(3, figsize=(8, 6))
ax1 = plt.subplot(3, 1, 1)
tw.plot(lattice_only=True, ax=ax1)
plt.plot(tw.s, tw.spin_dn_ddelta_x)
plt.plot(df.s, df.spin_dn_dpz_x)
plt.ylabel('dn_ddelta_x')
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
tw.plot(lattice_only=True, ax=ax2)
plt.plot(tw.s, tw.spin_dn_ddelta_y)
plt.plot(df.s, df.spin_dn_dpz_y)
plt.plot(df.s, dn_dy_ref_bmad, label='dn_y_ref_bmad')
plt.ylabel('dn_ddelta_y')
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
tw.plot(lattice_only=True, ax=ax3)
plt.plot(tw.s, tw.spin_dn_ddelta_z)
plt.plot(df.s, df.spin_dn_dpz_z)
plt.ylabel('dn_ddelta_z')
plt.xlabel('s [m]')


plt.show()
